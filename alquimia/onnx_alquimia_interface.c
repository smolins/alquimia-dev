/* -*-  mode: c; c-default-style: "google"; indent-tabs-mode: nil -*- */

/*
** Alquimia Copyright (c) 2013-2016, The Regents of the University of California,
** through Lawrence Berkeley National Laboratory (subject to receipt of any
** required approvals from the U.S. Dept. of Energy).  All rights reserved.
**
** Alquimia is available under a BSD license. See LICENSE.txt for more
** information.
**
** If you have questions about your rights to use or distribute this software,
** please contact Berkeley Lab's Technology Transfer and Intellectual Property
** Management at TTD@lbl.gov referring to Alquimia (LBNL Ref. 2013-119).
**
** NOTICE.  This software was developed under funding from the U.S. Department
** of Energy.  As such, the U.S. Government has been granted for itself and
** others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide
** license in the Software to reproduce, prepare derivative works, and perform
** publicly and display publicly.  Beginning five (5) years after the date
** permission to assert copyright is obtained from the U.S. Department of Energy,
** and subject to any subsequent five (5) year renewals, the U.S. Government is
** granted for itself and others acting on its behalf a paid-up, nonexclusive,
** irrevocable, worldwide license in the Software to reproduce, prepare derivative
** works, distribute copies to the public, perform publicly and display publicly,
** and to permit others to do so.
*/

/* **************************************************************************** 
**
** ONNX Alquimia Interface module
**
** Authors: 
**        Zhuolei Feng, Sergi Molins
**
** Notes:
**
**  * Public function call signatures, including intent, are dictated
**    by the alquimia API.
**
**  * alquimia data structures defined in the AlquimiaContainers_module
**    (alquimia_containers.h) are dictated by the alquimia API.
**
**  * All other function calls (e.g. involving ORT structures) are only
**    used here and not available in the interface
**
**  * It makes use of onnx_alquimia_config.h for reading the input file 
**    with information about the .onnx model file and metadata
**
** **************************************************************************** 
*/

#include "alquimia/onnx_alquimia_interface.h"

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ONNX Runtime's C header trips -Wpedantic in Alquimia's -Werror builds.
** Keep the diagnostic suppression scoped to the vendor include. */
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

#include <onnxruntime_c_api.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include "alquimia/alquimia_constants.h"
#include "alquimia/alquimia_util.h"
#include "alquimia/onnx_alquimia_config.h"

/* --------------------DATA STRUCTURES USED HERE ONLY ----------------------------*/
/* AlquimiaState fields that config mappings can name. The order must match
** kFieldTable below. */
typedef enum {
  kFieldWaterDensity,
  kFieldPorosity,
  kFieldTemperature,
  kFieldAqueousPressure,
  kFieldTotalMobile,
  kFieldTotalImmobile,
  kFieldTotalMobileImmobile,  /* derived: mobile + immobile, mol/L water */
  kFieldMineralVolumeFraction,
  kFieldMineralSpecificSurfaceArea,
  kFieldSurfaceSiteDensity,
  kFieldCationExchangeCapacity,
  kFieldGasConcentration,
  kFieldCount  /* number of fields; keep last */
} StateField;

/* Marks a table offset that does not apply to a field. */
#define kNoOffset SIZE_MAX

/* Everything the adapter needs to know about one StateField. */
typedef struct {
  const char *name;     /* alquimia_state value used in the JSON config */
  bool is_scalar;       /* scalar fields require alquimia_state_index 0 */
  size_t state_offset;  /* double (scalar) or AlquimiaVectorDouble in
                        ** AlquimiaState; kNoOffset for derived fields */
  size_t size_offset;   /* int in AlquimiaSizes; kNoOffset for scalars */
  size_t names_offset;  /* AlquimiaVectorString in AlquimiaProblemMetaData;
                        ** kNoOffset for scalars. Fields sharing a names
                        ** vector must share feature names at each index. */
} FieldInfo;

static const FieldInfo kFieldTable[] = {
  {"water_density", true, offsetof(AlquimiaState, water_density),
   kNoOffset, kNoOffset},
  {"porosity", true, offsetof(AlquimiaState, porosity),
   kNoOffset, kNoOffset},
  {"temperature", true, offsetof(AlquimiaState, temperature),
   kNoOffset, kNoOffset},
  {"aqueous_pressure", true, offsetof(AlquimiaState, aqueous_pressure),
   kNoOffset, kNoOffset},
  {"total_mobile", false, offsetof(AlquimiaState, total_mobile),
   offsetof(AlquimiaSizes, num_primary),
   offsetof(AlquimiaProblemMetaData, primary_names)},
  {"total_immobile", false, offsetof(AlquimiaState, total_immobile),
   offsetof(AlquimiaSizes, num_sorbed),
   offsetof(AlquimiaProblemMetaData, primary_names)},
  /* Also needs num_sorbed; see UpdateSizesForMapping. */
  {"total_mobile_immobile", false, kNoOffset,
   offsetof(AlquimiaSizes, num_primary),
   offsetof(AlquimiaProblemMetaData, primary_names)},
  {"mineral_volume_fraction", false,
   offsetof(AlquimiaState, mineral_volume_fraction),
   offsetof(AlquimiaSizes, num_minerals),
   offsetof(AlquimiaProblemMetaData, mineral_names)},
  {"mineral_specific_surface_area", false,
   offsetof(AlquimiaState, mineral_specific_surface_area),
   offsetof(AlquimiaSizes, num_minerals),
   offsetof(AlquimiaProblemMetaData, mineral_names)},
  {"surface_site_density", false,
   offsetof(AlquimiaState, surface_site_density),
   offsetof(AlquimiaSizes, num_surface_sites),
   offsetof(AlquimiaProblemMetaData, surface_site_names)},
  {"cation_exchange_capacity", false,
   offsetof(AlquimiaState, cation_exchange_capacity),
   offsetof(AlquimiaSizes, num_ion_exchange_sites),
   offsetof(AlquimiaProblemMetaData, ion_exchange_names)},
  {"gas_concentration", false, offsetof(AlquimiaState, gas_concentration),
   offsetof(AlquimiaSizes, num_gases),
   offsetof(AlquimiaProblemMetaData, gas_names)},
};

/* Compile-time check that kFieldTable has one entry per StateField. */
typedef char kFieldTableMatchesEnum[
    (sizeof(kFieldTable) / sizeof(kFieldTable[0]) == kFieldCount) ? 1 : -1];

typedef struct {
  StateField alquimia_state;
  /* Index into the state vector; always 0 for scalar fields. */
  int alquimia_state_index;
  /* Points into OnnxEngineState.onnx_config, which lives as long as the
  ** engine. Not owned. */
  const char *feature;
} FeatureMapping;

/* The model's input or output tensors, in session order. */
typedef struct {
  size_t num_tensors;
  char **names;          /* allocated by the session allocator */
  size_t *num_dim;       /* 0 for a scalar tensor */
  int64_t **dim_values;  /* dynamic batch extent fixed to 1; NULL for scalars */
  size_t *total_size;    /* elements per tensor */
  double **data;         /* buffers wrapped by tensor[] */
  OrtValue **tensor;
} OrtTensor;

typedef struct {
  const OrtApi *g_ort;  /* global function table; never released */
  OrtEnv *env;
  OrtSessionOptions *session_options;
  OrtSession *session;
  OrtMemoryInfo *memory_info;
  OrtAllocator *allocator;
} Ort;

typedef struct {
  Ort ort;
  /* Retained for named JSON conditions used by ProcessCondition. */
  OnnxAlquimiaConfig onnx_config;
  /* Selects JSON-backed initialization instead of driver constraints. */
  bool hands_off;

  /* Element counts over all input / output tensors, flattened in order. */
  size_t total_flat_inputs;
  size_t total_flat_outputs;

  OrtTensor input_tensors;
  OrtTensor output_tensors;

  FeatureMapping *input_mappings;  /* Array of size total_flat_inputs */
  FeatureMapping *output_mappings; /* Array of size total_flat_outputs */
  /* Parallel lookup indicating that the paired phase is also an output. */
  bool *output_has_paired_mapping;
} OnnxEngineState;

/* ------------- HELPER and ONNX-RELATED FUNCTIONS USED HERE ONLY -----------------*/

/* Returns true for a NULL (success) OrtStatus. Otherwise records the ONNX
** Runtime error in alquimia_status, releases the OrtStatus, returns false. */
static bool CheckStatus(const OrtApi *g_ort, OrtStatus *status, AlquimiaEngineStatus *alquimia_status)
{
  if (status != NULL)
  {
    const char *msg = g_ort->GetErrorMessage(status);
    alquimia_status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(alquimia_status->message, kAlquimiaMaxStringLength, "ONNX Runtime Error: %s", msg);
    g_ort->ReleaseStatus(status);
    return false;
  }
  return true;
}

/* Returns the engine behind an engine-state handle, or NULL after recording
** an invalid-engine error (e.g. Shutdown called after a failed Setup). */
static OnnxEngineState *EngineFromHandle(
    void *onnx_engine_state,
    const char *caller,
    AlquimiaEngineStatus *status)
{
  if (onnx_engine_state == NULL || *(OnnxEngineState **)onnx_engine_state == NULL)
  {
    status->error = kAlquimiaErrorInvalidEngine;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Invalid ONNX engine state pointer in %s.", caller);
    return NULL;
  }
  return *(OnnxEngineState **)onnx_engine_state;
}

/* Returns the table entry for a field. */
static const FieldInfo *FieldInfoFor(StateField field)
{
  return &kFieldTable[field];
}

/* Looks up a config alquimia_state name. Returns false if it is unknown. */
static bool FindField(const char *name, StateField *field)
{
  size_t i;
  for (i = 0; i < kFieldCount; ++i)
  {
    if (strcmp(name, kFieldTable[i].name) == 0)
    {
      *field = (StateField)i;
      return true;
    }
  }
  return false;
}

/* True when two vector fields write their names to the same metadata vector
** (e.g. total_mobile and total_immobile both use primary_names). */
static bool SharesMetadataNames(StateField a, StateField b)
{
  return !FieldInfoFor(a)->is_scalar && !FieldInfoFor(b)->is_scalar &&
         FieldInfoFor(a)->names_offset == FieldInfoFor(b)->names_offset;
}

/* Returns the metadata name vector for a field, or NULL for scalars. */
static AlquimiaVectorString *MetadataNamesForMapping(
    AlquimiaProblemMetaData *meta_data,
    StateField field)
{
  size_t offset = FieldInfoFor(field)->names_offset;
  if (offset == kNoOffset)
  {
    return NULL;
  }
  return (AlquimiaVectorString *)((char *)meta_data + offset);
}

/* Copies a feature name into names->data[index], if that slot exists. */
static void StoreMetadataName(
    AlquimiaVectorString *names,
    int index,
    const char *value)
{
  if (names == NULL || names->data == NULL || index < 0 || index >= names->size)
  {
    return;
  }

  strncpy(names->data[index], value, kAlquimiaMaxStringLength - 1);
  names->data[index][kAlquimiaMaxStringLength - 1] = '\0';
}

/**
 * @brief Converts one config destination into a runtime mapping.
 * @param[in] alquimia_state Case-sensitive AlquimiaState variable name.
 * @param[in] alquimia_state_index Nonnegative state-vector index from the config.
 * @param[out] mapping Returns the runtime destination on success.
 * @param[out] status Returns an engine-integrity error on incompatibility.
 * @return True when the state variable and index can be represented safely.
 *
 * Scalar state variables require index zero. INT_MAX is rejected for vectors
 * so the later size calculation (index + 1) cannot overflow an int.
 */
static bool ParseConfigMapping(
    const char *alquimia_state,
    int alquimia_state_index,
    FeatureMapping *mapping,
    AlquimiaEngineStatus *status)
{
  bool is_scalar;

  if (!FindField(alquimia_state, &mapping->alquimia_state))
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Unsupported AlquimiaState variable '%s' in ONNX config.",
             alquimia_state);
    return false;
  }
  is_scalar = FieldInfoFor(mapping->alquimia_state)->is_scalar;
  if ((is_scalar && alquimia_state_index != 0) ||
      (!is_scalar && alquimia_state_index == INT_MAX))
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX config AlquimiaState index %d is incompatible with "
             "variable '%s'.", alquimia_state_index, alquimia_state);
    return false;
  }
  mapping->alquimia_state_index = alquimia_state_index;
  return true;
}

/* Raises *size to at least required_size. */
static void GrowSize(int *size, int required_size)
{
  if (*size < required_size)
  {
    *size = required_size;
  }
}

/**
 * @brief Expands the AlquimiaSizes entry required by one mapping.
 * @param[in] mapping Validated state destination.
 * @param[in,out] sizes Accumulator initialized to zero before the first mapping.
 *
 * Vector capacity is the highest mapped index plus one. Fields backed by the
 * same Alquimia dimension, such as the two mineral vectors, share one entry.
 */
static void UpdateSizesForMapping(
    const FeatureMapping *mapping,
    AlquimiaSizes *sizes)
{
  int required_size = mapping->alquimia_state_index + 1;
  size_t offset = FieldInfoFor(mapping->alquimia_state)->size_offset;

  if (offset == kNoOffset)
  {
    return;
  }
  GrowSize((int *)((char *)sizes + offset), required_size);
  /* The derived total reads both total_mobile and total_immobile. */
  if (mapping->alquimia_state == kFieldTotalMobileImmobile)
  {
    GrowSize(&sizes->num_sorbed, required_size);
  }
}

/* Converts (tensor name, element index) to an index into the flat mapping
** array: tensors are laid out one after another in session order, the same
** order the reaction step walks them. Fails for an unknown or duplicate
** tensor name, or an element index past the tensor's size. */
static bool FindFlatTensorElement(
    const char *tensor,
    size_t tensor_element_index,
    size_t num_tensors,
    char *const *names,
    const size_t *tensor_sizes,
    size_t *flat_index,
    AlquimiaEngineStatus *status)
{
  size_t i;
  size_t offset = 0;
  size_t matching_index = num_tensors;  /* num_tensors means "not found" */

  for (i = 0; i < num_tensors; ++i)
  {
    if (strcmp(tensor, names[i]) == 0)
    {
      if (matching_index != num_tensors)
      {
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength,
                 "ONNX model contains duplicate tensor name '%s'.", tensor);
        return false;
      }
      matching_index = i;
      *flat_index = offset + tensor_element_index;
    }
    offset += tensor_sizes[i];
  }
  if (matching_index == num_tensors)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX config references unknown tensor '%s'.", tensor);
    return false;
  }
  if (tensor_element_index >= tensor_sizes[matching_index])
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX config tensor element index %zu is out of range for "
             "tensor '%s' of size %zu.", tensor_element_index, tensor,
             tensor_sizes[matching_index]);
    return false;
  }
  return true;
}

/**
 * @brief Checks feature names across all mappings, inputs first, then outputs.
 * @param[in] onnx_state Engine whose input and output mappings are complete.
 * @param[out] status Returns an engine-integrity error for the first conflict.
 * @return True when the names are consistent.
 *
 * Feature names double as metadata names (GetProblemMetaData), so:
 * - a name may repeat only at the same state index, either for fields that
 *   share a metadata vector (e.g. total_mobile and total_immobile), or for
 *   the same field read by an input and written by an output;
 * - two different names may not target the same metadata vector and index.
 */
static bool ValidateFeatureNames(
    const OnnxEngineState *onnx_state,
    AlquimiaEngineStatus *status)
{
  size_t num_inputs = onnx_state->total_flat_inputs;
  size_t num_all = num_inputs + onnx_state->total_flat_outputs;
  size_t a;
  size_t b;

  for (b = 0; b < num_all; ++b)
  {
    const FeatureMapping *later = b < num_inputs
        ? &onnx_state->input_mappings[b]
        : &onnx_state->output_mappings[b - num_inputs];
    for (a = 0; a < b; ++a)
    {
      const FeatureMapping *earlier = a < num_inputs
          ? &onnx_state->input_mappings[a]
          : &onnx_state->output_mappings[a - num_inputs];
      bool same_index =
          earlier->alquimia_state_index == later->alquimia_state_index;
      bool same_field = earlier->alquimia_state == later->alquimia_state;
      bool shares = SharesMetadataNames(earlier->alquimia_state,
                                        later->alquimia_state);

      if (strcmp(earlier->feature, later->feature) == 0)
      {
        bool input_and_output = a < num_inputs && b >= num_inputs;
        bool allowed = same_index &&
                       (input_and_output ? (same_field || shares)
                                         : (!same_field && shares));
        if (!allowed)
        {
          status->error = kAlquimiaErrorEngineIntegrity;
          snprintf(status->message, kAlquimiaMaxStringLength,
                   "Duplicate ONNX %s feature name '%s'.",
                   b < num_inputs ? "input" : "output", later->feature);
          return false;
        }
      }
      else if (shares && same_index)
      {
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength,
                 "Conflicting ONNX feature names '%s' and '%s' for "
                 "AlquimiaState variable '%s' index %d.",
                 earlier->feature, later->feature,
                 FieldInfoFor(later->alquimia_state)->name,
                 later->alquimia_state_index);
        return false;
      }
    }
  }
  return true;
}

/**
 * @brief Resolves where a condition value is written, using the model inputs.
 * @param[in] onnx_state Engine with complete input mappings.
 * @param[in] item Condition item: feature name, optional state field, value.
 * @param[out] mapping Destination; mapping->feature is NULL when the model
 *        does not read this feature and the item should be ignored.
 * @param[out] status Returns an engine-integrity error for invalid items.
 * @return True when resolved or ignored.
 *
 * Rules:
 * - total_mobile_immobile is derived, so it can't be set directly; set
 *   total_mobile and/or total_immobile instead.
 * - A bare number (no state field) is only accepted when the name matches
 *   exactly one input and that input is not total_mobile_immobile; otherwise
 *   the split between fields would be a guess.
 * - An explicit field must be the input's field or share its metadata
 *   vector (e.g. total_immobile for a total_mobile input); the index is
 *   taken from the input.
 */
static bool ResolveConditionMapping(
    const OnnxEngineState *onnx_state,
    const OnnxAlquimiaConditionItem *item,
    FeatureMapping *mapping,
    AlquimiaEngineStatus *status)
{
  size_t i;
  const FeatureMapping *matched_input = NULL;
  mapping->feature = NULL;
  if (item->alquimia_state != NULL &&
      !ParseConfigMapping(item->alquimia_state, 0, mapping, status))
  {
    return false;
  }
  if (item->alquimia_state != NULL &&
      mapping->alquimia_state == kFieldTotalMobileImmobile)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Condition feature '%s': total_mobile_immobile is derived; "
             "specify total_mobile and/or total_immobile.", item->feature);
    return false;
  }
  for (i = 0; i < onnx_state->total_flat_inputs; ++i)
  {
    const FeatureMapping *input = &onnx_state->input_mappings[i];
    if (strcmp(input->feature, item->feature) != 0)
    {
      continue;
    }
    /* Bare number for a repeated name or a derived total: ambiguous. */
    if (item->alquimia_state == NULL &&
        (matched_input != NULL || input->alquimia_state == kFieldTotalMobileImmobile))
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Condition feature '%s' requires explicit state fields.", item->feature);
      return false;
    }
    matched_input = input;
  }
  if (matched_input == NULL)
  {
    return true;  /* the model does not read this feature */
  }
  if (item->alquimia_state == NULL)
  {
    *mapping = *matched_input;
    return true;
  }
  if (mapping->alquimia_state != matched_input->alquimia_state &&
      !SharesMetadataNames(mapping->alquimia_state,
                           matched_input->alquimia_state))
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Condition field '%s' is incompatible with feature '%s'.",
             item->alquimia_state, item->feature);
    return false;
  }
  mapping->alquimia_state_index = matched_input->alquimia_state_index;
  mapping->feature = matched_input->feature;
  return true;
}

/**
 * @brief Maps the config entries of one role (inputs or outputs) onto the
 *        flattened tensor elements.
 * @param[in] entries Config mappings for this role.
 * @param[in] num_entries Number of config mappings.
 * @param[in] tensors Inspected model tensors for this role.
 * @param[in] total_flat Total number of flattened tensor elements.
 * @param[out] mappings Array of @p total_flat runtime mappings to fill.
 * @param[in] is_output True for outputs (total_mobile_immobile is input-only).
 * @param[in,out] sizes Grown to fit every mapped state index.
 * @param[out] status Returns the first mapping error.
 * @return True when every tensor element is mapped exactly once.
 */
static bool BuildRoleMappings(
    const OnnxAlquimiaMapping *entries,
    size_t num_entries,
    const OrtTensor *tensors,
    size_t total_flat,
    FeatureMapping *mappings,
    bool is_output,
    AlquimiaSizes *sizes,
    AlquimiaEngineStatus *status)
{
  const char *kind = is_output ? "output" : "input";
  bool *seen = (bool *)calloc(total_flat, sizeof(bool));
  bool ok = false;
  size_t i;

  if (seen == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Memory allocation failed for ONNX config mappings.");
    return false;
  }

  for (i = 0; i < num_entries; ++i)
  {
    const OnnxAlquimiaMapping *entry = &entries[i];
    size_t flat_index = 0;

    if (!FindFlatTensorElement(entry->tensor, entry->tensor_element_index,
                               tensors->num_tensors, tensors->names,
                               tensors->total_size, &flat_index, status))
    {
      goto done;
    }
    if (seen[flat_index])
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Duplicate ONNX %s mapping for tensor '%s' element index %zu.",
               kind, entry->tensor, entry->tensor_element_index);
      goto done;
    }
    if (!ParseConfigMapping(entry->alquimia_state, entry->alquimia_state_index,
                            &mappings[flat_index], status))
    {
      goto done;
    }
    if (is_output &&
        mappings[flat_index].alquimia_state == kFieldTotalMobileImmobile)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "ONNX total_mobile_immobile is an input-only mapping.");
      goto done;
    }
    /* Not copied: onnx_config lives as long as the engine. */
    mappings[flat_index].feature = entry->feature;
    seen[flat_index] = true;
    UpdateSizesForMapping(&mappings[flat_index], sizes);
  }

  for (i = 0; i < total_flat; ++i)
  {
    if (!seen[i])
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "ONNX config does not map every %s tensor element index.", kind);
      goto done;
    }
  }
  ok = true;

done:
  free(seen);
  return ok;
}

/**
 * @brief Builds the flat input/output mappings and the state sizes.
 * @param[in,out] onnx_state Inspected model state; receives the mappings.
 * @param[out] sizes Dimensions derived from the highest mapped indices,
 *        including state fields that conditions initialize.
 * @param[out] status Returns allocation or validation errors. Partial
 *        allocations are released by shutdown.
 * @return True when every tensor element has exactly one valid mapping.
 */
static bool BuildConfigMappings(
    OnnxEngineState *onnx_state,
    AlquimiaSizes *sizes,
    AlquimiaEngineStatus *status)
{
  const OnnxAlquimiaConfig *config = &onnx_state->onnx_config;
  size_t i;
  size_t j;

  onnx_state->input_mappings = (FeatureMapping *)calloc(
      onnx_state->total_flat_inputs, sizeof(*onnx_state->input_mappings));
  onnx_state->output_mappings = (FeatureMapping *)calloc(
      onnx_state->total_flat_outputs, sizeof(*onnx_state->output_mappings));
  if (onnx_state->input_mappings == NULL || onnx_state->output_mappings == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Memory allocation failed for ONNX config mappings.");
    return false;
  }
  memset(sizes, 0, sizeof(*sizes));

  /* Coverage is checked before names, since names need every mapping. */
  if (!BuildRoleMappings(config->inputs, config->num_inputs,
                         &onnx_state->input_tensors,
                         onnx_state->total_flat_inputs,
                         onnx_state->input_mappings, false, sizes, status) ||
      !BuildRoleMappings(config->outputs, config->num_outputs,
                         &onnx_state->output_tensors,
                         onnx_state->total_flat_outputs,
                         onnx_state->output_mappings, true, sizes, status) ||
      !ValidateFeatureNames(onnx_state, status))
  {
    return false;
  }

  /* A condition may initialize a field the model does not read for a known
  ** feature (e.g. mineral_specific_surface_area when the model reads
  ** mineral_volume_fraction), so size the state for those too.
  ** ResolveConditionMapping rejects incompatible fields and ignores
  ** features the model does not use (mapping.feature == NULL). */
  for (i = 0; i < config->num_conditions; ++i)
  {
    const OnnxAlquimiaCondition *condition = &config->conditions[i];
    for (j = 0; j < condition->num_items; ++j)
    {
      FeatureMapping mapping;
      if (!ResolveConditionMapping(onnx_state, &condition->items[j], &mapping,
                                   status))
      {
        return false;
      }
      if (mapping.feature != NULL)
      {
        UpdateSizesForMapping(&mapping, sizes);
      }
    }
  }
  return true;
}

/* Liquid water per bulk volume, 1000 * porosity * saturation [L/m^3]: the
** factor between mol/L water and mol/m^3 bulk. Requires porosity and
** saturation in (0, 1]; returns 0 after recording an error otherwise. */
static double WaterVolumePerBulk(
    double porosity,
    const AlquimiaProperties *properties,
    AlquimiaEngineStatus *status)
{
  double water_volume;
  if (properties == NULL || !isfinite(porosity) || porosity <= 0.0 ||
      porosity > 1.0 || !isfinite(properties->saturation) ||
      properties->saturation <= 0.0 || properties->saturation > 1.0)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX concentration conversion requires properties and finite "
             "porosity and saturation in (0, 1].");
    return 0.0;
  }
  water_volume = 1000.0 * porosity * properties->saturation;
  if (!isfinite(water_volume) || water_volume <= 0.0)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Invalid ONNX liquid water volume for concentration conversion.");
    return 0.0;
  }
  return water_volume;
}

/**
 * @brief Returns the address of the state value a mapping refers to.
 * @param[in] state State to address. Not modified here; callers decide.
 * @param[in] mapping Validated field and zero-based vector index.
 * @param[in] action "access" or "write", used in the error message.
 * @param[out] status Returns an engine integrity error for NULL vector
 *        storage, an out-of-bounds index, or a derived field with no slot.
 * @return The value's address, or NULL after recording an error.
 */
static double *StateSlot(
    const AlquimiaState *state,
    FeatureMapping mapping,
    const char *action,
    AlquimiaEngineStatus *status)
{
  const FieldInfo *info = FieldInfoFor(mapping.alquimia_state);
  /* One function serves reads and writes, so const is dropped here (like
  ** strchr); GetAlquimiaValue only reads through the result. */
  char *base = (char *)state;
  const AlquimiaVectorDouble *vector;
  int index = mapping.alquimia_state_index;

  if (info->state_offset == kNoOffset)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Derived field '%s' has no state storage.", info->name);
    return NULL;
  }
  if (info->is_scalar)
  {
    return (double *)(base + info->state_offset);
  }
  vector = (const AlquimiaVectorDouble *)(base + info->state_offset);
  if (vector->data == NULL || index < 0 || index >= vector->size)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Out-of-bounds %s %s: index %d, size %d.", info->name, action,
             index, vector->size);
    return NULL;
  }
  return &vector->data[index];
}

/**
 * @brief Reads native state units, or derived total concentration in mol/L water.
 * @param[in] state State containing the model input value.
 * @param[in] properties Supplies saturation for total concentration conversion.
 * @param[in] mapping Validated field and zero-based vector index.
 * @param[out] status Returns an engine integrity error for invalid access.
 * @return The mapped value on success, or 0.0 after recording an error.
 */
static double GetAlquimiaValue(
    const AlquimiaState *state,
    const AlquimiaProperties *properties,
    FeatureMapping mapping,
    AlquimiaEngineStatus *status)
{
  const double *slot;

  /* Derived total = mobile + immobile, in mol/L water. */
  if (mapping.alquimia_state == kFieldTotalMobileImmobile)
  {
    double mobile;
    double immobile;
    double total;
    double water_volume = WaterVolumePerBulk(state->porosity, properties, status);
    if (status->error != kAlquimiaNoError)
    {
      return 0.0;
    }
    mapping.alquimia_state = kFieldTotalMobile;
    mobile = GetAlquimiaValue(state, properties, mapping, status);
    if (status->error != kAlquimiaNoError)
    {
      return 0.0;
    }
    mapping.alquimia_state = kFieldTotalImmobile;
    immobile = GetAlquimiaValue(state, properties, mapping, status);
    if (status->error != kAlquimiaNoError)
    {
      return 0.0;
    }
    total = mobile + immobile / water_volume;
    if (!isfinite(total))
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Non-finite ONNX total_mobile_immobile at index %d.",
               mapping.alquimia_state_index);
      return 0.0;
    }
    return total;
  }

  slot = StateSlot(state, mapping, "access", status);
  return slot != NULL ? *slot : 0.0;
}

/* Reads one model input. total_immobile is converted from mol/m^3 bulk to
** mol/L water, since tensors use mol/L for both phases. Non-finite inputs
** are an error. */
static double GetAlquimiaModelInput(
    const AlquimiaState *state,
    const AlquimiaProperties *properties,
    FeatureMapping mapping,
    AlquimiaEngineStatus *status)
{
  double value = GetAlquimiaValue(state, properties, mapping, status);
  if (status->error != kAlquimiaNoError)
  {
    return 0.0;
  }
  if (mapping.alquimia_state == kFieldTotalImmobile)
  {
    double water_volume = WaterVolumePerBulk(state->porosity, properties, status);
    if (status->error != kAlquimiaNoError)
    {
      return 0.0;
    }
    value /= water_volume;  /* [moles/m^3 bulk] -> [molarity] */
  }
  if (!isfinite(value))
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Non-finite ONNX input for feature '%s'.", mapping.feature);
    return 0.0;
  }
  return value;
}

/**
 * @brief Writes a native-unit value to a mapped AlquimiaState destination.
 * @param[in,out] state State that receives the native-unit value.
 * @param[in] mapping Validated field and zero-based vector index.
 * @param[in] value Value in the destination field's native units.
 * @param[out] status Returns an engine integrity error for invalid access.
 */
static void SetAlquimiaValue(
    AlquimiaState *state,
    FeatureMapping mapping,
    double value,
    AlquimiaEngineStatus *status)
{
  double *slot = StateSlot(state, mapping, "write", status);
  if (slot != NULL)
  {
    *slot = value;
  }
}

/* True when any of the mappings refers to (alquimia_state, index). */
static bool HasMapping(
    const FeatureMapping *mappings,
    size_t num_mappings,
    StateField alquimia_state,
    int alquimia_state_index)
{
  size_t i;

  for (i = 0; i < num_mappings; ++i)
  {
    if (mappings[i].alquimia_state == alquimia_state &&
        mappings[i].alquimia_state_index == alquimia_state_index)
    {
      return true;
    }
  }
  return false;
}

/* For each mobile/immobile output, records at setup whether the
** conservation step should be skipped: true when the model also outputs the
** other phase, or does not read total_mobile_immobile for that component. */
static bool BuildPairedOutputLookup(
    OnnxEngineState *onnx_state,
    AlquimiaEngineStatus *status)
{
  size_t i;

  onnx_state->output_has_paired_mapping = (bool *)calloc(
      onnx_state->total_flat_outputs,
      sizeof(*onnx_state->output_has_paired_mapping));
  if (onnx_state->output_has_paired_mapping == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Memory allocation failed for ONNX paired output lookup.");
    return false;
  }

  for (i = 0; i < onnx_state->total_flat_outputs; ++i)
  {
    const FeatureMapping *mapping = &onnx_state->output_mappings[i];
    StateField paired_state;

    if (mapping->alquimia_state == kFieldTotalMobile)
    {
      paired_state = kFieldTotalImmobile;
    }
    else if (mapping->alquimia_state == kFieldTotalImmobile)
    {
      paired_state = kFieldTotalMobile;
    }
    else
    {
      continue;
    }
    onnx_state->output_has_paired_mapping[i] =
        !HasMapping(onnx_state->input_mappings, onnx_state->total_flat_inputs,
                    kFieldTotalMobileImmobile, mapping->alquimia_state_index) ||
        HasMapping(onnx_state->output_mappings, onnx_state->total_flat_outputs,
                   paired_state, mapping->alquimia_state_index);
  }
  return true;
}

/**
 * @brief Writes one model output and preserves a paired component total.
 * @param[in,out] state State receiving the model output and any paired update.
 * @param[in] properties Supplies liquid saturation.
 * @param[in] old_porosity Porosity before inference.
 * @param[in] new_porosity Final porosity after all mapped outputs.
 * @param[in] mapping Validated model-output destination.
 * @param[in] has_paired_mapping Whether conservation of the paired phase is skipped.
 * @param[in] value Model output; mobile and immobile values are mol/L water.
 * @param[out] status Returns mapped-state access errors.
 *
 * When a model maps total_mobile_immobile[i] as an input and outputs only one
 * of total_mobile[i] or total_immobile[i], the paired value conserves bulk
 * inventory: 1000 * porosity * saturation * mobile + immobile. Explicit model
 * outputs for both values, or components without total_mobile_immobile[i],
 * write only the mapped output.
 */
static void SetAlquimiaModelOutput(
    AlquimiaState *state,
    const AlquimiaProperties *properties,
    double old_porosity,
    double new_porosity,
    FeatureMapping mapping,
    bool has_paired_mapping,
    double value,
    AlquimiaEngineStatus *status)
{
  int index = mapping.alquimia_state_index;
  FeatureMapping paired_mapping;  /* the other phase of the same component */
  double ncomp;
  double mapped_value;
  double paired_value;
  double old_water_volume;
  double new_water_volume;

  /* Immobile outputs are mol/L water; convert to mol/m^3 bulk. */
  if (mapping.alquimia_state == kFieldTotalImmobile)
  {
    new_water_volume = WaterVolumePerBulk(new_porosity, properties, status);
    if (status->error != kAlquimiaNoError)
    {
      return;
    }
    value *= new_water_volume;
    if (!isfinite(value))
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Non-finite ONNX immobile output conversion at index %d.", index);
      return;
    }
  }

  paired_mapping = mapping;

  /* No paired slot in the state: just write the output. */
  if (mapping.alquimia_state == kFieldTotalMobile)
  {
    paired_mapping.alquimia_state = kFieldTotalImmobile;
    if (state->total_immobile.data == NULL || index < 0 ||
        index >= state->total_immobile.size)
    {
      SetAlquimiaValue(state, mapping, value, status);
      return;
    }
  }
  else if (mapping.alquimia_state == kFieldTotalImmobile)
  {
    paired_mapping.alquimia_state = kFieldTotalMobile;
    if (state->total_mobile.data == NULL || index < 0 ||
        index >= state->total_mobile.size)
    {
      SetAlquimiaValue(state, mapping, value, status);
      return;
    }
  }
  else
  {
    SetAlquimiaValue(state, mapping, value, status);
    return;
  }

  /* The model outputs both phases (or has no derived total input): no
  ** conservation step. */
  if (has_paired_mapping)
  {
    SetAlquimiaValue(state, mapping, value, status);
    return;
  }

  old_water_volume = WaterVolumePerBulk(old_porosity, properties, status);
  if (status->error != kAlquimiaNoError)
  {
    return;
  }
  if (mapping.alquimia_state == kFieldTotalMobile)
  {
    new_water_volume = WaterVolumePerBulk(new_porosity, properties, status);
    if (status->error != kAlquimiaNoError)
    {
      return;
    }
  }
  mapped_value = GetAlquimiaValue(state, properties, mapping, status);
  if (status->error != kAlquimiaNoError)
  {
    return;
  }
  paired_value = GetAlquimiaValue(state, properties, paired_mapping, status);
  if (status->error != kAlquimiaNoError)
  {
    return;
  }

  /* Conserve the bulk inventory ncomp [mol/m^3 bulk]:
  **   water_volume_old * mobile_old + immobile_old
  **     = water_volume_new * mobile_new + immobile_new
  ** with mobile in mol/L and immobile in mol/m^3 bulk. */
  if (mapping.alquimia_state == kFieldTotalMobile)
  {
    ncomp = old_water_volume * mapped_value + paired_value;
    paired_value = ncomp - new_water_volume * value;  /* new immobile */
  }
  else
  {
    ncomp = old_water_volume * paired_value + mapped_value;
    paired_value = (ncomp - value) / new_water_volume;  /* new mobile */
  }
  if (!isfinite(ncomp) || !isfinite(paired_value))
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Non-finite ONNX mobile/immobile conservation at index %d.", index);
    return;
  }

  SetAlquimiaValue(state, mapping, value, status);
  if (status->error != kAlquimiaNoError)
  {
    return;
  }
  SetAlquimiaValue(state, paired_mapping, paired_value, status);
}

/* Releases one tensor collection; works on partially filled ones. */
static void ReleaseOrtTensors(OrtTensor *tensors, Ort *ort)
{
  size_t i;

  if (tensors->tensor != NULL)
  {
    for (i = 0; i < tensors->num_tensors; ++i)
    {
      if (tensors->tensor[i] != NULL)
      {
        ort->g_ort->ReleaseValue(tensors->tensor[i]);
      }
    }
    free(tensors->tensor);
  }
  if (tensors->data != NULL)
  {
    for (i = 0; i < tensors->num_tensors; ++i)
    {
      free(tensors->data[i]);
    }
    free(tensors->data);
  }
  if (tensors->dim_values != NULL)
  {
    for (i = 0; i < tensors->num_tensors; ++i)
    {
      free(tensors->dim_values[i]);
    }
    free(tensors->dim_values);
  }
  if(tensors->num_dim != NULL)
  {
    free(tensors->num_dim); 
  }
  if(tensors->total_size)
  {
    free(tensors->total_size);
  }

  /* Names came from the session allocator. */
  if (ort->allocator != NULL)
  {
    if (tensors->names != NULL)
    {
      for (i = 0; i < tensors->num_tensors; ++i)
      {
        if (tensors->names[i] != NULL)
        {
          ort->allocator->Free(ort->allocator, tensors->names[i]);
        }
      }
      free(tensors->names);
    }
  }
  memset(tensors, 0, sizeof(*tensors));
}

/* Releases a partially initialized engine via shutdown, using a scratch
** status so the caller's status keeps the setup error. */
static void CleanupOnSetupFailure(OnnxEngineState **onnx_state)
{
  char temp_message[kAlquimiaMaxStringLength];
  AlquimiaEngineStatus temp_status = {0};
  temp_status.message = temp_message;
  onnx_alquimia_shutdown(onnx_state, &temp_status);
}

/**
 * @brief Records one model tensor's shape and wraps a buffer for it.
 * @param[in] api ONNX Runtime API.
 * @param[in] memory_info CPU memory info used to create the OrtValue.
 * @param[in] type_info Type info of the tensor; owned by the caller.
 * @param[in] kind "input" or "output", used in error messages.
 * @param[in,out] tensors Collection whose entry @p i is filled in.
 * @param[in] i Index of the tensor in session order.
 * @param[out] status Returns errors; partial allocations are left in
 *        @p tensors for ReleaseOrtTensors.
 * @return True on success.
 *
 * Only double tensors are supported. A dynamic (<= 0) leading extent of a
 * rank >= 2 tensor is the batch dimension and is fixed to one, because the
 * interface runs one state at a time; any other dynamic extent is rejected.
 * A scalar (rank 0) keeps a NULL dimension array and holds one element.
 */
static bool InspectTensor(
    const OrtApi *api,
    const OrtMemoryInfo *memory_info,
    const OrtTypeInfo *type_info,
    const char *kind,
    OrtTensor *tensors,
    size_t i,
    AlquimiaEngineStatus *status)
{
  const OrtTensorTypeAndShapeInfo *tensor_info = NULL;
  ONNXTensorElementDataType element_type;
  size_t num_dim = 0;
  size_t total_size = 1;
  size_t j;
  int64_t *dims = NULL;

  if (!CheckStatus(api, api->CastTypeInfoToTensorInfo(type_info, &tensor_info), status) ||
      !CheckStatus(api, api->GetTensorElementType(tensor_info, &element_type), status))
  {
    return false;
  }
  if (element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX %s tensor '%s' must have double elements.",
             kind, tensors->names[i]);
    return false;
  }
  if (!CheckStatus(api, api->GetDimensionsCount(tensor_info, &num_dim), status))
  {
    return false;
  }
  tensors->num_dim[i] = num_dim;
  if (num_dim > 0)
  {
    dims = (int64_t *)calloc(num_dim, sizeof(int64_t));
    tensors->dim_values[i] = dims;
    if (dims == NULL)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Memory allocation failed for ONNX %s tensor dimensions.", kind);
      return false;
    }
    if (!CheckStatus(api, api->GetDimensions(tensor_info, dims, num_dim), status))
    {
      return false;
    }
  }

  for (j = 0; j < num_dim; ++j)
  {
    if (dims[j] <= 0)
    {
      if (j != 0 || num_dim == 1)
      {
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength,
                 "ONNX %s tensor '%s' has an unsupported dynamic extent.",
                 kind, tensors->names[i]);
        return false;
      }
      dims[j] = 1;  /* batch size */
    }
    total_size *= (size_t)dims[j];
  }
  tensors->total_size[i] = total_size;

  tensors->data[i] = (double *)calloc(total_size, sizeof(double));
  if (tensors->data[i] == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Memory allocation failed for ONNX %s tensor data.", kind);
    return false;
  }
  /* The OrtValue wraps data[i]; ONNX Runtime does not own that buffer. */
  return CheckStatus(api, api->CreateTensorWithDataAsOrtValue(
                              memory_info, tensors->data[i],
                              total_size * sizeof(double), dims, num_dim,
                              ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
                              &tensors->tensor[i]),
                     status);
}

/**
 * @brief Inspects all model inputs or all model outputs.
 * @param[in] ort Initialized session, memory info and allocator.
 * @param[in] is_input True for inputs, false for outputs.
 * @param[out] tensors Receives names, shapes, buffers and OrtValues.
 * @param[out] total_flat Receives the summed element count of all tensors.
 * @param[out] status Returns errors; partial allocations are left in
 *        @p tensors for ReleaseOrtTensors.
 * @return True on success.
 */
static bool InspectTensors(
    const Ort *ort,
    bool is_input,
    OrtTensor *tensors,
    size_t *total_flat,
    AlquimiaEngineStatus *status)
{
  const OrtApi *api = ort->g_ort;
  const char *kind = is_input ? "input" : "output";
  size_t num_tensors = 0;
  size_t i;

  if (!CheckStatus(api, is_input
                            ? api->SessionGetInputCount(ort->session, &num_tensors)
                            : api->SessionGetOutputCount(ort->session, &num_tensors),
                   status))
  {
    return false;
  }
  if (num_tensors == 0)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Model has 0 inputs or outputs (no %s tensors).", kind);
    return false;
  }

  tensors->num_tensors = num_tensors;
  tensors->names = (char **)calloc(num_tensors, sizeof(char *));
  tensors->num_dim = (size_t *)calloc(num_tensors, sizeof(size_t));
  tensors->dim_values = (int64_t **)calloc(num_tensors, sizeof(int64_t *));
  tensors->total_size = (size_t *)calloc(num_tensors, sizeof(size_t));
  tensors->data = (double **)calloc(num_tensors, sizeof(double *));
  tensors->tensor = (OrtValue **)calloc(num_tensors, sizeof(OrtValue *));
  if (tensors->names == NULL || tensors->num_dim == NULL ||
      tensors->dim_values == NULL || tensors->total_size == NULL ||
      tensors->data == NULL || tensors->tensor == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Memory allocation failed for ONNX %s tensor arrays.", kind);
    return false;
  }

  *total_flat = 0;
  for (i = 0; i < num_tensors; ++i)
  {
    OrtTypeInfo *type_info = NULL;
    bool ok;

    /* Names are allocated with the session allocator (freed in
    ** ReleaseOrtTensors). */
    if (!CheckStatus(api, is_input
                              ? api->SessionGetInputName(ort->session, i, ort->allocator, &tensors->names[i])
                              : api->SessionGetOutputName(ort->session, i, ort->allocator, &tensors->names[i]),
                     status) ||
        !CheckStatus(api, is_input
                              ? api->SessionGetInputTypeInfo(ort->session, i, &type_info)
                              : api->SessionGetOutputTypeInfo(ort->session, i, &type_info),
                     status))
    {
      return false;
    }
    ok = InspectTensor(api, ort->memory_info, type_info, kind, tensors, i,
                       status);
    api->ReleaseTypeInfo(type_info);
    if (!ok)
    {
      return false;
    }
    *total_flat += tensors->total_size[i];
  }
  return true;
}

/* ------------------------ ALQUIMIA FUNCTIONS -------------------------------*/

/**
 * @brief Initializes an ONNX engine from a versioned JSON sidecar config.
 * @param[in] input_filename JSON config path; relative model paths resolve from its
 *        directory.
 * @param[in] hands_off Selects named JSON conditions for initialization.
 * @param[in,out] onnx_engine_state Address that receives the initialized engine.
 * @param[in,out] sizes Returns state-vector sizes derived from explicit mappings.
 * @param[out] functionality Returns the ONNX adapter capability flags.
 * @param[out] status Returns setup errors without being overwritten by cleanup.
 *
 * The destination engine pointer is set to NULL first and stays NULL on every
 * failure; all failures release partial state through one cleanup path. The
 * parsed config stays engine-owned so named conditions are available to
 * ProcessCondition.
 */
void onnx_alquimia_setup(
    const char *input_filename,
    bool hands_off,
    void *onnx_engine_state,
    AlquimiaSizes *sizes,
    AlquimiaEngineFunctionality *functionality,
    AlquimiaEngineStatus *status)
{
  OnnxEngineState *onnx_state;
  Ort *ort;
  const OrtApi *api;
  FILE *f;

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  *(OnnxEngineState **)onnx_engine_state = NULL;

  onnx_state = (OnnxEngineState *)calloc(1, sizeof(OnnxEngineState));
  if (onnx_state == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for OnnxEngineState.");
    return;
  }
  ort = &onnx_state->ort;

  /* JSON config: model path, tensor-to-state mappings, conditions. */
  if (!OnnxAlquimiaLoadConfig(input_filename, &onnx_state->onnx_config,
                              status->message, kAlquimiaMaxStringLength))
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    goto fail;
  }
  onnx_state->hands_off = hands_off;
  if (hands_off && onnx_state->onnx_config.num_conditions == 0)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX hands-off setup requires at least one JSON condition.");
    goto fail;
  }
  /* Checked here for a clearer message than ONNX Runtime's load error. */
  f = fopen(onnx_state->onnx_config.model_path, "rb");
  if (f == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX model file not found: %s",
             onnx_state->onnx_config.model_path);
    goto fail;
  }
  fclose(f);

  api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (api == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Failed to load ONNX Runtime API.");
    goto fail;
  }
  ort->g_ort = api;

  if (!CheckStatus(api, api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                       "onnx_alquimia_engine", &ort->env),
                   status) ||
      !CheckStatus(api, api->CreateSessionOptions(&ort->session_options),
                   status) ||
      !CheckStatus(api, api->CreateSession(ort->env,
                                           onnx_state->onnx_config.model_path,
                                           ort->session_options, &ort->session),
                   status) ||
      !CheckStatus(api, api->CreateCpuMemoryInfo(OrtArenaAllocator,
                                                 OrtMemTypeDefault,
                                                 &ort->memory_info),
                   status) ||
      !CheckStatus(api, api->CreateAllocator(ort->session, ort->memory_info,
                                             &ort->allocator),
                   status) ||
      !InspectTensors(ort, true, &onnx_state->input_tensors,
                      &onnx_state->total_flat_inputs, status) ||
      !InspectTensors(ort, false, &onnx_state->output_tensors,
                      &onnx_state->total_flat_outputs, status) ||
      !BuildConfigMappings(onnx_state, sizes, status) ||
      !BuildPairedOutputLookup(onnx_state, status))
  {
    goto fail;
  }

  /* Each reaction step mutates the reusable tensors and backing buffers in
  ** OnnxEngineState, so shared-state calls are not thread-safe. MPI-only use
  ** remains safe because each process owns its engine state. */
  functionality->thread_safe = false;
  functionality->temperature_dependent =
      HasMapping(onnx_state->input_mappings, onnx_state->total_flat_inputs,
                 kFieldTemperature, 0);
  functionality->pressure_dependent =
      HasMapping(onnx_state->input_mappings, onnx_state->total_flat_inputs,
                 kFieldAqueousPressure, 0);
  functionality->porosity_update =
      HasMapping(onnx_state->output_mappings, onnx_state->total_flat_outputs,
                 kFieldPorosity, 0);
  functionality->operator_splitting = true;
  functionality->global_implicit = false;
  functionality->index_base = 0;

  *(OnnxEngineState **)onnx_engine_state = onnx_state;
  return;

fail:
  /* Shutdown releases whatever was created so far (it skips NULL members). */
  CleanupOnSetupFailure(&onnx_state);
}

/**
 * @brief Releases all adapter and ONNX Runtime resources.
 * @param[in,out] onnx_engine_state Address of the engine pointer created by
 *        setup; set to NULL on success.
 * @param[out] status Returns an invalid-engine error for a NULL engine.
 *
 * Works on partially initialized engines too (setup failures use it), since
 * it skips members that were never created.
 */
void onnx_alquimia_shutdown(
    void *onnx_engine_state,
    AlquimiaEngineStatus *status)
{
  OnnxEngineState *onnx_state;

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  onnx_state = EngineFromHandle(onnx_engine_state, "Shutdown", status);
  if (onnx_state == NULL)
  {
    return;
  }

  if (onnx_state->ort.g_ort != NULL)
  {
    /* Mapping feature names point into onnx_config (freed below). */
    free(onnx_state->input_mappings);
    onnx_state->input_mappings = NULL;
    free(onnx_state->output_mappings);
    onnx_state->output_mappings = NULL;
    free(onnx_state->output_has_paired_mapping);
    onnx_state->output_has_paired_mapping = NULL;

    ReleaseOrtTensors(&onnx_state->input_tensors, &onnx_state->ort);
    ReleaseOrtTensors(&onnx_state->output_tensors, &onnx_state->ort);

    /* Release core ONNX Runtime objects in reverse initialization order. */
    if (onnx_state->ort.allocator != NULL)
    {
      onnx_state->ort.g_ort->ReleaseAllocator(onnx_state->ort.allocator);
      onnx_state->ort.allocator = NULL;
    }
    if (onnx_state->ort.memory_info != NULL)
    {
      onnx_state->ort.g_ort->ReleaseMemoryInfo(
          onnx_state->ort.memory_info);
      onnx_state->ort.memory_info = NULL;
    }
    if (onnx_state->ort.session != NULL)
    {
      onnx_state->ort.g_ort->ReleaseSession(onnx_state->ort.session);
      onnx_state->ort.session = NULL;
    }
    if (onnx_state->ort.session_options != NULL)
    {
      onnx_state->ort.g_ort->ReleaseSessionOptions(
          onnx_state->ort.session_options);
      onnx_state->ort.session_options = NULL;
    }
    if (onnx_state->ort.env != NULL)
    {
      onnx_state->ort.g_ort->ReleaseEnv(onnx_state->ort.env);
      onnx_state->ort.env = NULL;
    }
  }
  OnnxAlquimiaFreeConfig(&onnx_state->onnx_config);
  free(onnx_state);
  *(OnnxEngineState **)onnx_engine_state = NULL;
}

/**
 * @brief Applies JSON or driver conditions to config-mapped model inputs.
 * @param[in] onnx_engine_state Address of an initialized engine pointer.
 * @param[in] condition Condition whose exact name selects JSON values in hands-off
 *        mode, or whose aqueous constraints supply values in normal mode.
 * @param[in] properties [Unused]; conditions assign native state fields directly.
 * @param[in,out] state State receiving values for matching input feature names.
 * @param[in] aux_data [Unused] by the ONNX adapter.
 * @param[out] status Returns invalid-engine or mapped-state access errors.
 *
 * Hands-off mode ignores driver constraint values and requires an exact JSON
 * condition-name match. Normal mode preserves the generic constraint behavior:
 * absent constraints leave their mapped state values unchanged.
 */
void onnx_alquimia_processcondition(
    void *onnx_engine_state,
    AlquimiaGeochemicalCondition *condition,
    AlquimiaProperties *properties,
    AlquimiaState *state,
    AlquimiaAuxiliaryData *aux_data,
    AlquimiaEngineStatus *status)
{
  OnnxEngineState *onnx_state;
  size_t k;
  int c;

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  (void)aux_data;
  (void)properties;

  onnx_state = EngineFromHandle(onnx_engine_state, "ProcessCondition", status);
  if (onnx_state == NULL)
  {
    return;
  }


  if (state == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid state pointer in ProcessCondition.");
    return;
  }

  /* Hands-off mode: values come from the JSON condition with this name. */
  if (onnx_state->hands_off)
  {
    const OnnxAlquimiaCondition *matching_condition = NULL;
    size_t condition_index;
    
    if (condition != NULL && condition->name != NULL)
    {
      for (condition_index = 0;
           condition_index < onnx_state->onnx_config.num_conditions;
           ++condition_index)
      {
        const OnnxAlquimiaCondition *current_condition =
            &onnx_state->onnx_config.conditions[condition_index];
        if (strcmp(current_condition->name, condition->name) == 0)
        {
          matching_condition = current_condition;
          break;
        }
      }
    }
    if (matching_condition == NULL)
    {
      status->error = kAlquimiaErrorUnknownConstraintName;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Unknown ONNX JSON condition name '%s'.",
               condition != NULL && condition->name != NULL
                   ? condition->name
                   : "");
      return;
    }

    for (k = 0; k < matching_condition->num_items; ++k)
    {
      const OnnxAlquimiaConditionItem *item = &matching_condition->items[k];
      FeatureMapping mapping;
      if (!ResolveConditionMapping(onnx_state, item, &mapping, status))
      {
        return;
      }
      if (mapping.feature != NULL)
      {
        SetAlquimiaValue(state, mapping, item->value, status);
        if (status->error != kAlquimiaNoError)
        {
          return;
        }
      }
    }
    return;
  }

  /* Driver mode: an absent condition preserves the existing state. */
  if (condition == NULL || condition->aqueous_constraints.data == NULL)
  {
    return;
  }

  /* Driver constraints carry only a name and a value, with no state field,
  ** so each is resolved as scalar shorthand: a name the model maps to more
  ** than one field (e.g. mobile and immobile) is rejected before anything is
  ** written for it, and names the model does not read are ignored. */
  for (c = 0; c < condition->aqueous_constraints.size; ++c)
  {
    const AlquimiaAqueousConstraint *constraint =
        &condition->aqueous_constraints.data[c];
    OnnxAlquimiaConditionItem item = {0};
    FeatureMapping mapping;

    item.feature = constraint->primary_species_name;
    item.value = constraint->value;
    if (!ResolveConditionMapping(onnx_state, &item, &mapping, status))
    {
      return;
    }
    if (mapping.feature != NULL)
    {
      SetAlquimiaValue(state, mapping, item.value, status);
      if (status->error != kAlquimiaNoError)
      {
        return;
      }
    }
  }
}

/**
 * @brief Runs one operator-split ONNX inference and routes its outputs.
 * @param[in,out] onnx_engine_state Address of an initialized engine pointer.
 * @param[in] delta_t [Unused]; the model receives only explicitly mapped state data.
 * @param[in] properties Supplies saturation for concentration unit conversions.
 * @param[in,out] state Supplies mapped inputs and receives mapped outputs.
 * @param[in] aux_data [Unused] by the ONNX adapter.
 * @param[in] natural_id [Unused] by the ONNX adapter.
 * @param[out] status Returns state-access or ONNX Runtime errors.
 *
 * The engine reuses mutable tensor buffers allocated during setup. Calls that
 * share one engine instance are therefore not thread-safe.
 */
void onnx_alquimia_reactionstepoperatorsplit(
    void *onnx_engine_state,
    double delta_t,
    AlquimiaProperties *properties,
    AlquimiaState *state,
    AlquimiaAuxiliaryData *aux_data,
    int natural_id,
    AlquimiaEngineStatus *status)
{
  OnnxEngineState *onnx_state;
  OrtStatus *ort_status;
  size_t i;
  size_t k;
  size_t flat_index;
  double old_porosity;
  double new_porosity;

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  (void)delta_t;
  (void)aux_data;
  (void)natural_id;

  onnx_state = EngineFromHandle(onnx_engine_state, "ReactionStepOperatorSplit", status);
  if (onnx_state == NULL)
  {
    return;
  }



  old_porosity = state->porosity;
  new_porosity = old_porosity;

  /* Fill the input buffers (wrapped by the input OrtValues). */
  flat_index = 0;
  for (i = 0; i < onnx_state->input_tensors.num_tensors; ++i)
  {
    for (k = 0; k < onnx_state->input_tensors.total_size[i]; ++k, ++flat_index)
    {
      onnx_state->input_tensors.data[i][k] = GetAlquimiaModelInput(
          state, properties, onnx_state->input_mappings[flat_index], status);
      if (status->error != kAlquimiaNoError)
      {
        return;
      }
    }
  }

  ort_status = onnx_state->ort.g_ort->Run(
      onnx_state->ort.session,
      NULL, /* RunOptions */
      (const char *const *)onnx_state->input_tensors.names,
      (const OrtValue *const *)onnx_state->input_tensors.tensor,
      onnx_state->input_tensors.num_tensors,
      (const char *const *)onnx_state->output_tensors.names,
      onnx_state->output_tensors.num_tensors,
      onnx_state->output_tensors.tensor);
  if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
  {
    return;
  }

  /* The output OrtValues wrap output_tensors.data, so Run wrote the results
  ** there. First pass: reject non-finite values and find the new porosity,
  ** which the immobile conversions need regardless of output order. */
  flat_index = 0;
  for (i = 0; i < onnx_state->output_tensors.num_tensors; ++i)
  {
    for (k = 0; k < onnx_state->output_tensors.total_size[i]; ++k, ++flat_index)
    {
      double value = onnx_state->output_tensors.data[i][k];
      if (!isfinite(value))
      {
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength,
                 "Non-finite ONNX output for feature '%s'.",
                 onnx_state->output_mappings[flat_index].feature);
        return;
      }
      if (onnx_state->output_mappings[flat_index].alquimia_state ==
          kFieldPorosity)
      {
        new_porosity = value;
      }
    }
  }

  /* Second pass: write the outputs to the state. */
  flat_index = 0;
  for (i = 0; i < onnx_state->output_tensors.num_tensors; ++i)
  {
    for (k = 0; k < onnx_state->output_tensors.total_size[i]; ++k, ++flat_index)
    {
      SetAlquimiaModelOutput(
          state, properties, old_porosity, new_porosity,
          onnx_state->output_mappings[flat_index],
          onnx_state->output_has_paired_mapping[flat_index],
          onnx_state->output_tensors.data[i][k], status);
      if (status->error != kAlquimiaNoError)
      {
        return;
      }
    }
  }
}

/**
 * @brief Implements the generic auxiliary-output hook as a successful no-op.
 *
 * Version 1 ONNX configs map only AlquimiaState fields and define no
 * auxiliary outputs.
 */
void onnx_alquimia_getauxiliaryoutput(
    void *onnx_engine_state,
    AlquimiaProperties *properties,
    AlquimiaState *state,
    AlquimiaAuxiliaryData *aux_data,
    AlquimiaAuxiliaryOutputData *aux_out,
    AlquimiaEngineStatus *status)
{
  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  if (EngineFromHandle(onnx_engine_state, "GetAuxiliaryOutput", status) == NULL)
  {
    return;
  }

  (void)properties;
  (void)state;
  (void)aux_data;
  (void)aux_out;
}

/**
 * @brief Copies config input/output feature names into problem metadata vectors.
 * @param[in] onnx_engine_state Address of an initialized engine pointer.
 * @param[in,out] meta_data Metadata storage allocated from the setup-derived sizes.
 * @param[out] status Returns invalid-engine or invalid-destination errors.
 *
 * Fields that share an Alquimia metadata vector were checked for conflicting
 * names during setup, so each destination index has one representable name.
 */
void onnx_alquimia_getproblemmetadata(
    void *onnx_engine_state,
    AlquimiaProblemMetaData *meta_data,
    AlquimiaEngineStatus *status)
{
  OnnxEngineState *onnx_state;
  size_t i;

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  onnx_state = EngineFromHandle(onnx_engine_state, "GetProblemMetaData", status);
  if (onnx_state == NULL)
  {
    return;
  }


  for (i = 0; i < onnx_state->total_flat_inputs; ++i)
  {
    FeatureMapping mapping = onnx_state->input_mappings[i];
    AlquimiaVectorString *metadata_name = MetadataNamesForMapping(
        meta_data, mapping.alquimia_state);
    StoreMetadataName(metadata_name, mapping.alquimia_state_index, mapping.feature);
  }
  for (i = 0; i < onnx_state->total_flat_outputs; ++i)
  {
    FeatureMapping mapping = onnx_state->output_mappings[i];
    AlquimiaVectorString *metadata_name = MetadataNamesForMapping(
        meta_data, mapping.alquimia_state);
    StoreMetadataName(metadata_name, mapping.alquimia_state_index,
                      mapping.feature);
  }
}
