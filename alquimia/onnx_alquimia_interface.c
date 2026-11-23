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

typedef struct {
  /* Dynamic input/output info */
  /* Align with the num_inputs/outputs in onnx_config */
  size_t num_tensors;
  /* Align with the tensor in onnx_config */
  char **names;
  /* Number of dimensions for each input/output tensor
  ** 0 for a scalar tensor
  */
  size_t *num_dim;
  /* [batch size, input/output features] / [input/output features]
  ** batch size == -1 means it's dynamic
  ** Considering the architecture of Alquimia, batch size = 1
  */
  int64_t **dim_values; 
  /* Each input/output tensor has their own size */
  size_t *total_size; 
  /* The real numbers for each input/output tensor. */
  double **data;
  OrtValue **tensor;
} OrtTensor;

typedef struct {
  /* OnnxRuntime API */
  /* Const pointer to the global OrtApi function table. Do NOT release. */
  const OrtApi *g_ort;
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

  /* Flatten non-sequential input/output tensors into 1-dimensional arrays. */
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

/**
 * @brief Converts an ONNX Runtime status into an Alquimia engine status.
 * @param[in] g_ort ONNX Runtime API used to inspect and release @p status.
 * @param[in] status ONNX status returned by an API call; NULL indicates success.
 * @param[in,out] alquimia_status Destination for the translated error and message.
 * @return True on success. On failure, releases @p status, records an engine
 *         integrity error, and returns false.
 */
static bool CheckStatus(const OrtApi *g_ort, OrtStatus *status, AlquimiaEngineStatus *alquimia_status)
{
  /* A non-NULL OrtStatus indicates an API failure; extract the error, 
  ** populate our status, and explicitly release the status object. 
  */
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

/**
 * @brief Copies a config feature name into an allocated Alquimia name vector.
 * @param[in,out] names Destination name vector.
 * @param[in] index Zero-based destination index.
 * @param[in] value Null-terminated feature name to copy.
 */
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

/**
 * @brief Resolves a tensor element index to the flat runtime mapping index.
 * @param[in] tensor Case-sensitive tensor name from the onnx_config.
 * @param[in] tensor_element_index Zero-based flattened index within that tensor.
 * @param[in] num_tensors Number of model tensors in @p names and @p tensor_sizes.
 * @param[in] names ONNX Runtime tensor names in session order.
 * @param[in] tensor_sizes Flattened element counts in session order.
 * @param[out] flat_index Returns the offset into the combined mapping array.
 * @param[out] status Returns an error for unknown, duplicate, or undersized tensors.
 * @return True when exactly one tensor matches and the index is in range.
 *
 * Reaction steps iterate tensors in session order, so this same prefix-sum
 * layout must be used while constructing the mapping arrays.
 */
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
  /* Initialize sentinel value to track if the target tensor is found. */
  size_t matching_index = num_tensors;

  /* Check if there duplicate values in JSON */
  for (i = 0; i < num_tensors; ++i)
  {
    if (strcmp(tensor, names[i]) == 0)
    {
      /* If matching_index was already set, a duplicate name exists. */
      if (matching_index != num_tensors)
      {
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength,
                 "ONNX model contains duplicate tensor name '%s'.", tensor);
        return false;
      }
      /* Find the tensor customized_tensor_element[0] for the first time */
      matching_index = i;
      /* Calculate the absolute flattened index for this specific element. */
      *flat_index = offset + tensor_element_index;
    }
    /* Accumulate tensor offset to track the starting bound of subsequent tensors. */
    offset += tensor_sizes[i];
  }
  /* Reject unknown tensors not present in the model metadata. */
  if (matching_index == num_tensors)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX config references unknown tensor '%s'.", tensor);
    return false;
  }
  /* Guard against out-of-bounds element indexing within the matched tensor. */
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
 * @brief Allows a shared feature name only across compatible fields at one index.
 * @param[in] mappings Existing flattened input or output mappings.
 * @param[in] mapping Candidate state destination.
 * @param[in] seen Marks the mappings that have already been populated.
 * @param[in] num_mappings Number of entries in the mapping and seen arrays.
 * @param[in] feature Case-sensitive feature name to validate.
 * @param[in] mapping_kind Mapping direction ("input" or "output") for error messages.
 * @param[in] allow_same_state True when comparing an output against input mappings.
 * @param[out] status Returns an engine-integrity error for a duplicate name.
 * @return True when the feature name does not conflict with existing mappings.
 */
static bool ValidateUniqueFeature(
    const FeatureMapping *mappings,
    const FeatureMapping *mapping,
    const bool *seen,
    size_t num_mappings,
    const char *feature,
    const char *mapping_kind,
    bool allow_same_state,
    AlquimiaEngineStatus *status)
{
  size_t i;

  for (i = 0; i < num_mappings; ++i)
  {
    if (seen[i] && strcmp(mappings[i].feature, feature) == 0)
    {
      bool same_index =
          mapping->alquimia_state_index == mappings[i].alquimia_state_index;
      bool same_state =
          mapping->alquimia_state == mappings[i].alquimia_state;

      /* A name may repeat at the same index across fields that share a
      ** metadata vector (e.g. total_mobile and total_immobile). */
      if (same_index &&
          ((allow_same_state && same_state) ||
           (!same_state && SharesMetadataNames(mapping->alquimia_state,
                                               mappings[i].alquimia_state))))
      {
        continue;
      }
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Duplicate ONNX %s feature name '%s'.", mapping_kind, feature);
      return false;
    }
  }
  return true;
}

/**
 * @brief Resolves condition destinations independently of the model's input fields.
 * @param[in] onnx_state Inspected model state and destination for runtime mappings.
 * @param[in] item Candidate JSON condition item to resolve.
 * @param[out] mapping Resolved destination mapping. A NULL feature denotes an ignored extra feature.
 * @param[out] status Returns an engine-integrity error for invalid or incompatible fields.
 * @return True when the condition is successfully resolved or safely ignored.
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

  /* 
   * Reject explicit initialization of a derived total.
   * 
   * You cannot directly assign a value to 'total_mobile_immobile' because it is a 
   * derived sum, not a fundamental storage slot. If a single total value is provided, 
   * the system cannot arbitrarily guess how to partition that mass between the 
   * fluid phase (total_mobile) and the solid phase (total_immobile).
   * The user must explicitly target the underlying mobile or immobile slots.
   */
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
    /* Skip if the model input name (e.g., "H+") doesn't match the JSON item name (e.g., "Zn"). */
    if (strcmp(input->feature, item->feature) != 0)
    {
      continue;
    }

    /* 
     * DEFENSE AGAINST AMBIGUOUS SCALAR SHORTHAND
     * `item->alquimia_state == NULL` means the JSON provided a bare number without 
     * a physical state label. (e.g., JSON has `"Zn": 0.001` instead of `"Zn": {"total_mobile": 0.001}`)
     * 
     * If the user used this shorthand, we MUST block two highly dangerous scenarios:
     */
    if (item->alquimia_state == NULL &&
        (matched_input != NULL || input->alquimia_state == kFieldTotalMobileImmobile))
    {
      /* 
       * FAILURE SCENARIO 1: `matched_input != NULL` (Duplicate Name Ambiguity)
       * - Model Mapping : Has TWO inputs named "Zn" (e.g., Index 0 is aqueous, Index 1 is solid).
       * - JSON Condition: "Zn": 0.001
       * - What happens  : On the first loop, `matched_input` becomes the aqueous "Zn". On the second loop, 
       *                   we find the solid "Zn". Since `matched_input != NULL`, we catch the ambiguity.
       * - Why it fails  : The system refuses to blindly guess which "Zn" gets the 0.001.
       * 
       * FAILURE SCENARIO 2: `... == kFieldTotalMobileImmobile` (Derived Quantity Ambiguity)
       * - Model Mapping : Has ONE input named "Zn", defined as TOTAL_MOBILE_IMMOBILE.
       * - JSON Condition: "Zn": 0.001
       * - What happens  : The condition matches, but it hits this rule.
       * - Why it fails  : You cannot write directly to a "Total". The system needs to know 
       *                   how to split that 0.001 into the underlying mobile/immobile arrays.
       * 
       * THE FIX FOR BOTH: The user is forced to write explicit JSON:
       * "Zn": { "total_mobile": 0.001 }
       */
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Condition feature '%s' requires explicit state fields.", item->feature);
      return false;
    }
    /* Record the safe match. If the loop finds another input with the exact same name, 
     * this `matched_input` variable will trigger Failure Scenario 1. */
    matched_input = input;
  }
  /* If the JSON feature does not exist in the ONNX model, safely ignore it. */
  if (matched_input == NULL)
  {
    return true;
  }
  /* 
   * If the user used the safe scalar shorthand, inherit the exact model mapping.
   * We only proceed to verify category compatibility if the user explicitly 
   * provided an alquimia_state in the JSON.
   */
  if (item->alquimia_state == NULL)
  {
    *mapping = *matched_input;
    return true;
  }
  /* 
   * CATEGORY COMPATIBILITY VERIFICATION
   * At this point, the user has explicitly specified a state field. We must ensure 
   * they are not mixing up physical phases (e.g., trying to write a mineral volume 
   * fraction into an aqueous concentration slot).
   * 
   * Allow flexibility (the requested state doesn't have to perfectly match the
   * model's expected state) ONLY IF both states share a metadata name vector
   * (e.g. total_mobile/total_immobile, or the two mineral fields).
   */
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
 * @brief Rejects a feature name that conflicts at one metadata destination.
 * @param[in] mappings Existing flattened input or output mappings.
 * @param[in] seen Marks mappings that have already been populated.
 * @param[in] num_mappings Number of entries in the mapping and seen arrays.
 * @param[in] mapping Parsed runtime destination for the candidate mapping.
 * @param[in] config_mapping Candidate config mapping and feature name.
 * @param[out] status Returns an engine-integrity error for a conflict.
 * @return True when the candidate is compatible with all populated mappings.
 */
static bool ValidateConsistentFeatureName(
    const FeatureMapping *mappings,
    const bool *seen,
    size_t num_mappings,
    const FeatureMapping *mapping,
    const OnnxAlquimiaMapping *config_mapping,
    AlquimiaEngineStatus *status)
{
  size_t i;

  for (i = 0; i < num_mappings; ++i)
  {
    const FeatureMapping *other = &mappings[i];
    if (seen[i] &&
        SharesMetadataNames(other->alquimia_state, mapping->alquimia_state) &&
        other->alquimia_state_index == mapping->alquimia_state_index &&
        strcmp(other->feature, config_mapping->feature) != 0)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Conflicting ONNX feature names '%s' and '%s' for "
               "AlquimiaState variable '%s' index %d.",
               other->feature, config_mapping->feature,
               config_mapping->alquimia_state,
               config_mapping->alquimia_state_index);
      return false;
    }
  }
  return true;
}

/**
 * @brief Builds complete flat input/output mappings from the parsed onnx_config.
 * @param[in,out] onnx_state Inspected model state and destination for runtime mappings.
 * @param[in,out] sizes Returns dimensions derived from the highest mapped indices.
 * @param[out] status Returns allocation or semantic-validation errors.
 * @return True only when every tensor element index has exactly one mapping.
 *
 * Input feature names are copied before the onnx_config is released. The seen
 * arrays enforce complete coverage and prevent duplicate tensor indices;
 * name-category checks preserve the metadata representation invariant.
 */
static bool BuildConfigMappings(
    OnnxEngineState *onnx_state,
    AlquimiaSizes *sizes,
    AlquimiaEngineStatus *status)
{
  /* Check if there is duplicate value in the JSON 
  ** They are for the 1-dimension array
  ** The input/output mapping are all 1-dimension arrays
  */
  bool *input_seen;
  bool *output_seen;
  size_t i;

  /* Memory Allocation */
  onnx_state->input_mappings = (FeatureMapping *)calloc(
      onnx_state->total_flat_inputs, sizeof(*onnx_state->input_mappings));
  onnx_state->output_mappings = (FeatureMapping *)calloc(
      onnx_state->total_flat_outputs, sizeof(*onnx_state->output_mappings));
  input_seen = (bool *)calloc(onnx_state->total_flat_inputs, sizeof(bool));
  output_seen = (bool *)calloc(onnx_state->total_flat_outputs, sizeof(bool));
  if (onnx_state->input_mappings == NULL ||
      onnx_state->output_mappings == NULL ||
      input_seen == NULL || output_seen == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Memory allocation failed for ONNX config mappings.");
    free(input_seen);
    free(output_seen);
    return false;
  }
  /* Initialize AlquimiaSize */
  memset(sizes, 0, sizeof(*sizes));

  /* For every input tensor */
  for (i = 0; i < onnx_state->onnx_config.num_inputs; ++i)
  {
    /* This points to the real OnnxEngine->onnx_config.inputMapping */
    const OnnxAlquimiaMapping *config_input =
        &onnx_state->onnx_config.inputs[i];
    FeatureMapping *mapping;
    size_t flat_index;

    /* Map the tensor element index into the 1-dimensional flattened array. */
    if (!FindFlatTensorElement(config_input->tensor,
                               config_input->tensor_element_index,
                               onnx_state->input_tensors.num_tensors,
                               onnx_state->input_tensors.names,
                               onnx_state->input_tensors.total_size,
                               &flat_index, status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    /* Find the duplicate value */
    if (input_seen[flat_index])
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Duplicate ONNX input mapping for tensor '%s' element index %zu.",
               config_input->tensor, config_input->tensor_element_index);
      free(input_seen);
      free(output_seen);
      return false;
    }
    mapping = &onnx_state->input_mappings[flat_index];
    /* Parse the AlquimiaState value to the input mapping */
    if (!ParseConfigMapping(config_input->alquimia_state,
                            config_input->alquimia_state_index, mapping,
                            status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    /* Check the duplicate input feature */
    if (!ValidateUniqueFeature(
            onnx_state->input_mappings, mapping, input_seen,
            onnx_state->total_flat_inputs, config_input->feature, "input",
            false, status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    /* Check if there are conflicted input features that target the same
    ** AlquimiaState metadata name.
    */
    if (!ValidateConsistentFeatureName(
            onnx_state->input_mappings, input_seen,
            onnx_state->total_flat_inputs, mapping, config_input, status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    /* Copy the feature name */
    /* Not copied: onnx_config lives as long as the engine. */
    mapping->feature = config_input->feature;
    /* Find the input tensor element for the first time */
    input_seen[flat_index] = true;
    /* Update the respective AlquimiaSize when finding new input tensor element */
    UpdateSizesForMapping(mapping, sizes);
  }

  /* For every output tensor */
  for (i = 0; i < onnx_state->onnx_config.num_outputs; ++i)
  {
    /* This points to the real OnnxEngine->onnx_config.outputMapping */
    const OnnxAlquimiaMapping *config_output =
        &onnx_state->onnx_config.outputs[i];
    FeatureMapping *mapping;
    size_t flat_index;

    /* Map the tensor element index into the 1-dimensional flattened array. */
    if (!FindFlatTensorElement(config_output->tensor,
                               config_output->tensor_element_index,
                               onnx_state->output_tensors.num_tensors,
                               onnx_state->output_tensors.names,
                               onnx_state->output_tensors.total_size,
                               &flat_index, status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    /* Find the duplicate value */
    if (output_seen[flat_index])
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Duplicate ONNX output mapping for tensor '%s' element index %zu.",
               config_output->tensor, config_output->tensor_element_index);
      free(input_seen);
      free(output_seen);
      return false;
    }
    mapping = &onnx_state->output_mappings[flat_index];
    /* Parse the AlquimiaState value to the output mapping */
    if (!ParseConfigMapping(config_output->alquimia_state,
                            config_output->alquimia_state_index, mapping,
                            status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    /* total_mobile_immobile can only be the input alquimia state schema */
    if (mapping->alquimia_state == kFieldTotalMobileImmobile)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "ONNX total_mobile_immobile is an input-only mapping.");
      free(input_seen);
      free(output_seen);
      return false;
    }
    /* Check the duplicate output feature */
    if (!ValidateUniqueFeature(
            onnx_state->input_mappings, mapping, input_seen,
            onnx_state->total_flat_inputs, config_output->feature, "output",
            true, status) ||
        !ValidateUniqueFeature(
            onnx_state->output_mappings, mapping, output_seen,
            onnx_state->total_flat_outputs, config_output->feature, "output",
            false, status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    /* Compare between output mapping and input mappings */
    if (!ValidateConsistentFeatureName(
            onnx_state->input_mappings, input_seen,
            onnx_state->total_flat_inputs, mapping, config_output, status) ||
        /* Compare between output mapping and output mappings */
        !ValidateConsistentFeatureName(
            onnx_state->output_mappings, output_seen,
            onnx_state->total_flat_outputs, mapping, config_output, status))
    {
      free(input_seen);
      free(output_seen);
      return false;
    }
    /* Not copied: onnx_config lives as long as the engine. */
    mapping->feature = config_output->feature;
    /* Find the output tensor element for the first time */
    output_seen[flat_index] = true;
    /* Update the respective AlquimiaSize when finding new output tensor element */
    UpdateSizesForMapping(mapping, sizes);
  }

  for (i = 0; i < onnx_state->total_flat_inputs; ++i)
  {
    if (!input_seen[i])
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "ONNX config does not map every input tensor element index.");
      free(input_seen);
      free(output_seen);
      return false;
    }
  }
  for (i = 0; i < onnx_state->total_flat_outputs; ++i)
  {
    if (!output_seen[i])
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "ONNX config does not map every output tensor element index.");
      free(input_seen);
      free(output_seen);
      return false;
    }
  }

  free(input_seen);
  free(output_seen);
  
  /* 
   * Expand state allocation sizes to accommodate valid, extra condition fields.
   *
   * A user's JSON condition may initialize a related state field (e.g., 
   * mineral_specific_surface_area) for a known component, even if the ONNX 
   * model only explicitly requires a different field (e.g., mineral_volume_fraction) 
   * for that same component.
   *
   * ResolveConditionMapping validates these category-compatible fields and 
   * safely ignores completely unrelated features (returning mapping.feature == NULL).
   * For valid matches, UpdateSizesForMapping ensures the Alquimia state vector 
   * allocates enough memory slots to store these initialized values.
   */
  for (i = 0; i < onnx_state->onnx_config.num_conditions; ++i)
  {
    const OnnxAlquimiaCondition *condition = &onnx_state->onnx_config.conditions[i];
    size_t j;
    for (j = 0; j < condition->num_items; ++j)
    {
      FeatureMapping mapping;

      /* Validate the condition item against known model inputs */
      if (!ResolveConditionMapping(onnx_state, &condition->items[j], &mapping, status))
      {
        return false; /* Fails immediately on ambiguous or physically incompatible fields */
      }

      /* If the feature is relevant to the model (not safely ignored), update memory sizes */
      if (mapping.feature != NULL)
      {
        UpdateSizesForMapping(&mapping, sizes);
      }
    }
  }
  return true;
}

/**
 * @brief Calculates the volume of liquid water per unit of bulk material volume.
 * @param[in] porosity The porosity of the porous medium in the range (0, 1].
 * @param[in] properties Supplies saturation for total concentration conversion.
 * @param[out] status Engine status populated if input bounds are violated.
 * @return The volume of water in liters per cubic meter of bulk material (L/m³).
 */
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

/**
 * @brief Reads a model input, converting immobile inventory to mol/L water.
 * @param[in] state State containing the model input value.
 * @param[in] properties Supplies saturation for water volume conversion.
 * @param[in] mapping Validated destination field and zero-based vector index.
 * @param[out] status Returns an engine integrity error for a non-finite model input, 
 *        or propagates errors from value extraction and volume calculation.
 * @return The standardized input value on success, or 0.0 after recording an error.
 */
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

/**
 * @brief Checks whether the model explicitly reads one state element.
 * @param[in] onnx_state Engine containing the flattened input mappings.
 * @param[in] alquimia_state State field to find.
 * @param[in] alquimia_state_index Zero-based state-vector index to find.
 * @return True when an input tensor element maps to the requested state element.
 */
static bool HasMappedInput(
    const OnnxEngineState *onnx_state,
    StateField alquimia_state,
    int alquimia_state_index)
{
  size_t i;

  for (i = 0; i < onnx_state->total_flat_inputs; ++i)
  {
    const FeatureMapping *mapping = &onnx_state->input_mappings[i];
    if (mapping->alquimia_state == alquimia_state &&
        mapping->alquimia_state_index == alquimia_state_index)
    {
      return true;
    }
  }
  return false;
}

/**
 * @brief Checks whether the model explicitly outputs one state element.
 * @param[in] onnx_state Engine containing the flattened output mappings.
 * @param[in] alquimia_state State field to find.
 * @param[in] alquimia_state_index Zero-based state-vector index to find.
 * @return True when an output tensor element maps to the requested state element.
 */
static bool HasMappedOutput(
    const OnnxEngineState *onnx_state,
    StateField alquimia_state,
    int alquimia_state_index)
{
  size_t i;

  for (i = 0; i < onnx_state->total_flat_outputs; ++i)
  {
    const FeatureMapping *mapping = &onnx_state->output_mappings[i];
    if (mapping->alquimia_state == alquimia_state &&
        mapping->alquimia_state_index == alquimia_state_index)
    {
      return true;
    }
  }
  return false;
}

/**
 * @brief Caches whether each output explicitly maps its paired phase.
 * @param[in,out] onnx_state Engine containing the completed output mappings.
 * @param[out] status Returns an allocation error for the lookup array.
 * @return True when the output-aligned lookup is initialized.
 *
 * The one-time setup scan keeps paired-output detection out of the repeated
 * inference path.
 */
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
        !HasMappedInput(onnx_state, kFieldTotalMobileImmobile,
                        mapping->alquimia_state_index) ||
        HasMappedOutput(onnx_state, paired_state,
                        mapping->alquimia_state_index);
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
  /* mapping = mobile, paired_mapping = immobile
  ** mapping = immobile, paired_mapping = mobile
  */
  FeatureMapping paired_mapping;
  double ncomp;
  double mapped_value;
  double paired_value;
  double old_water_volume;
  double new_water_volume;

  /* Normalize every immobile output before assignment or conservation,
   * including explicit paired outputs and fields without a mobile counterpart. */
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

  /* The model runs the inference for mobile */
  if (mapping.alquimia_state == kFieldTotalMobile)
  {
    paired_mapping.alquimia_state = kFieldTotalImmobile;

    /* Invalid immobile */
    if (state->total_immobile.data == NULL || index < 0 ||
        index >= state->total_immobile.size)
    {
      SetAlquimiaValue(state, mapping, value, status);
      return;
    }
  } /* The model runs the inference for immobile */
  else if (mapping.alquimia_state == kFieldTotalImmobile)
  {
    paired_mapping.alquimia_state = kFieldTotalMobile;
    if (state->total_mobile.data == NULL || index < 0 ||
        index >= state->total_mobile.size)
    {
      SetAlquimiaValue(state, mapping, value, status);
      return;
    }
  } /* Neither mobile nor immobile */
  else
  {
    SetAlquimiaValue(state, mapping, value, status);
    return;
  }

  /* Explicit model outputs for both phases remain authoritative. */
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

  if (mapping.alquimia_state == kFieldTotalMobile)
  {
    /* paired_mapping is immobile 
    ** mapping is mobile [molarity]
    ** Old total mobile + immobile = new mobile + immobile
    ** The inference is for mobile [molarity]
    ** We need to balance the total mobile [molarity] and immobile [moles/m^3 bulk]
    ** ncomp [moles/m^3 bulk]
    */
    ncomp = old_water_volume * mapped_value + paired_value;
    /* Unit: [moles/m^3 bulk]*/
    paired_value = ncomp - new_water_volume * value;
  }
  else
  {
    /* paired_mapping is mobile
    ** mapping is immobile [moles/m^3 bulk]
    ** Old total mobile + immobile = new mobile + immobile
    ** The immobile output has already been converted to [moles/m^3 bulk].
    ** We need to balance the total mobile [molarity] and immobile [moles/m^3 bulk]
    ** ncomp [moles/m^3 bulk]
    */
    ncomp = old_water_volume * paired_value + mapped_value;
    /* Unit: [molarity] */
    paired_value = (ncomp - value) / new_water_volume;
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

/**
 * @brief Releases one input or output tensor collection.
 * @param[in,out] tensors Tensor metadata and buffers to release.
 * @param[in] ort ONNX Runtime objects used to release names and OrtValue objects.
 */
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

  /* ONNX Runtime allocates tensor names with the OrtAllocator. */
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

/**
 * @brief Releases a partially initialized engine without replacing setup error details.
 * @param[in,out] onnx_state Address of the local engine pointer. Shutdown releases all
 *        resources initialized so far and sets this pointer to NULL.
 *
 * A stack-backed temporary status satisfies the shutdown contract while the
 * caller's original AlquimiaEngineStatus retains the setup failure.
 */
static void CleanupOnSetupFailure(OnnxEngineState **onnx_state)
{
  char temp_message[kAlquimiaMaxStringLength];
  AlquimiaEngineStatus temp_status = {0};
  temp_status.message = temp_message;
  onnx_alquimia_shutdown(onnx_state, &temp_status);
}

/* ------------------------ ALQUIMIA FUNCTIONS -------------------------------*/

/**
 * @brief Initializes an ONNX engine from a versioned JSON sidecar onnx_config.
 * @param[in] input_filename JSON config path; relative model paths resolve from its
 *        directory.
 * @param[in] hands_off Selects named JSON conditions for initialization.
 * @param[in,out] onnx_engine_state Address that receives the initialized engine.
 * @param[in,out] sizes Returns state-vector sizes derived from explicit mappings.
 * @param[out] functionality Returns the ONNX adapter capability flags.
 * @param[out] status Returns setup errors without being overwritten by cleanup.
 *
 * The destination engine pointer is set to NULL before resource acquisition and
 * remains NULL on every failure. Parsed config storage remains engine-owned so
 * named conditions are available to ProcessCondition.
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
  OrtStatus *ort_status;
  FILE *f;
  size_t i, j;

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  if (onnx_engine_state == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Invalid ONNX engine state destination in setup.");
    return;
  }
  *(OnnxEngineState **)onnx_engine_state = NULL;

  onnx_state = (OnnxEngineState *)calloc(1, sizeof(OnnxEngineState));
  if (onnx_state == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for OnnxEngineState.");
    return;
  }
  /* Read the JSON file that serves as input
  ** and provides the metadata mapping model to alquimia struct */
  if (!OnnxAlquimiaLoadConfig(
          input_filename, &onnx_state->onnx_config, status->message,
          kAlquimiaMaxStringLength))
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    free(onnx_state);
    return;
  }
  onnx_state->hands_off = hands_off;
  if (hands_off && onnx_state->onnx_config.num_conditions == 0)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX hands-off setup requires at least one JSON condition.");
    OnnxAlquimiaFreeConfig(&onnx_state->onnx_config);
    free(onnx_state);
    return;
  }
  f = fopen(onnx_state->onnx_config.model_path, "rb");
  if (f == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "ONNX model file not found: %s",
             onnx_state->onnx_config.model_path);
    OnnxAlquimiaFreeConfig(&onnx_state->onnx_config);
    free(onnx_state);
    return;
  }
  fclose(f);

  onnx_state->ort.g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!onnx_state->ort.g_ort)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Failed to load ONNX Runtime API.");
    OnnxAlquimiaFreeConfig(&onnx_state->onnx_config);
    free(onnx_state);
    return;
  }

  /* Read the ONNX file that contains the model */
  ort_status = onnx_state->ort.g_ort->CreateEnv(
      ORT_LOGGING_LEVEL_WARNING, "onnx_alquimia_engine",
      &onnx_state->ort.env);
  if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
  {
    OnnxAlquimiaFreeConfig(&onnx_state->onnx_config);
    free(onnx_state);
    return;
  }

  ort_status = onnx_state->ort.g_ort->CreateSessionOptions(
      &onnx_state->ort.session_options);
  if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
  {
    onnx_state->ort.g_ort->ReleaseEnv(onnx_state->ort.env);
    OnnxAlquimiaFreeConfig(&onnx_state->onnx_config);
    free(onnx_state);
    return;
  }

  ort_status = onnx_state->ort.g_ort->CreateSession(
      onnx_state->ort.env, onnx_state->onnx_config.model_path,
      onnx_state->ort.session_options, &onnx_state->ort.session);
  if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
  {
    onnx_state->ort.g_ort->ReleaseSessionOptions(
        onnx_state->ort.session_options);
    onnx_state->ort.g_ort->ReleaseEnv(onnx_state->ort.env);
    OnnxAlquimiaFreeConfig(&onnx_state->onnx_config);
    free(onnx_state);
    return;
  }

  ort_status = onnx_state->ort.g_ort->CreateCpuMemoryInfo(
      OrtArenaAllocator, OrtMemTypeDefault,
      &onnx_state->ort.memory_info);
  if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
  {
    CleanupOnSetupFailure(&onnx_state);
    return;
  }

  ort_status = onnx_state->ort.g_ort->CreateAllocator(
      onnx_state->ort.session, onnx_state->ort.memory_info,
      &onnx_state->ort.allocator);
  if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
  {
    CleanupOnSetupFailure(&onnx_state);
    return;
  }

  /* Query input/output tensor count */
  size_t num_inputs = 0;
  ort_status = onnx_state->ort.g_ort->SessionGetInputCount(
      onnx_state->ort.session, &num_inputs);
  if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
  {
    CleanupOnSetupFailure(&onnx_state);
    return;
  }
  onnx_state->input_tensors.num_tensors = num_inputs;

  size_t num_outputs = 0;
  ort_status = onnx_state->ort.g_ort->SessionGetOutputCount(
      onnx_state->ort.session, &num_outputs);
  if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
  {
    CleanupOnSetupFailure(&onnx_state);
    return;
  }
  onnx_state->output_tensors.num_tensors = num_outputs;

  /* Guard against empty inputs or outputs */
  if (num_inputs == 0 || num_outputs == 0)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Model has 0 inputs or outputs (inputs: %d, outputs: %d).", (int)num_inputs, (int)num_outputs);
    CleanupOnSetupFailure(&onnx_state);
    return;
  }

  /* Allocate outer arrays for inputs */
  onnx_state->input_tensors.names =
      (char **)calloc(num_inputs, sizeof(char *));
  onnx_state->input_tensors.num_dim =
      (size_t *)calloc(num_inputs, sizeof(size_t));
  onnx_state->input_tensors.dim_values =
      (int64_t **)calloc(num_inputs, sizeof(int64_t *));
  onnx_state->input_tensors.total_size =
      (size_t *)calloc(num_inputs, sizeof(size_t));
  onnx_state->input_tensors.data =
      (double **)calloc(num_inputs, sizeof(double *));
  onnx_state->input_tensors.tensor =
      (OrtValue **)calloc(num_inputs, sizeof(OrtValue *));

  /* Guard */
  if (onnx_state->input_tensors.names == NULL ||
      onnx_state->input_tensors.num_dim == NULL ||
      onnx_state->input_tensors.dim_values == NULL ||
      onnx_state->input_tensors.total_size == NULL ||
      onnx_state->input_tensors.data == NULL ||
      onnx_state->input_tensors.tensor == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for input container arrays.");
    CleanupOnSetupFailure(&onnx_state);
    return;
  }

  /* Parse input names and dimensions */
  for (i = 0; i < num_inputs; ++i)
  {
    char *name = NULL;
    /* Get the names of the input tensor[i] */
    ort_status = onnx_state->ort.g_ort->SessionGetInputName(
        onnx_state->ort.session, i, onnx_state->ort.allocator, &name);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    onnx_state->input_tensors.names[i] = name;

    OrtTypeInfo *ort_type_info = NULL;  /* Released by ReleaseTypeInfo */
    ort_status = onnx_state->ort.g_ort->SessionGetInputTypeInfo(
        onnx_state->ort.session, i, &ort_type_info);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    const OrtTensorTypeAndShapeInfo *tensor_info = NULL;  /* Do not free this value. It will be valid until ort_type_info is freed */
    /* Check if the input data type is double */
    ONNXTensorElementDataType onnx_element_type; /* No need to free. enum */
    ort_status = onnx_state->ort.g_ort->CastTypeInfoToTensorInfo(
        ort_type_info, &tensor_info);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Get the input data type */
    ort_status = onnx_state->ort.g_ort->GetTensorElementType(
        tensor_info, &onnx_element_type);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Check if the input data type is double */
    if (onnx_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "ONNX input tensor '%s' must have double elements.",
               onnx_state->input_tensors.names[i]);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Record the dimension number */
    size_t dim_count = 0;
    /* Get the number of the dimensions */
    ort_status = onnx_state->ort.g_ort->GetDimensionsCount(
        tensor_info, &dim_count);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    /* Preserve the model rank. A scalar has zero dimensions and keeps the
    ** NULL dimension array initialized by calloc. */
    onnx_state->input_tensors.num_dim[i] = dim_count;
    if (dim_count > 0)
    {
      onnx_state->input_tensors.dim_values[i] =
          (int64_t *)calloc(dim_count, sizeof(int64_t));
      if (onnx_state->input_tensors.dim_values[i] == NULL)
      {
        onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for input_dim_values.");
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
      /* Get the dimensions */
      ort_status = onnx_state->ort.g_ort->GetDimensions(
          tensor_info, onnx_state->input_tensors.dim_values[i], dim_count);
      if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
      {
        onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
    }

    /* The supported trained models use an unknown leading batch extent. The
    ** interface operates on one state at a time, so that batch is fixed to
    ** one. Other dynamic extents cannot be sized safely at setup. */
    /* Calculate the total input numbers for each tensor */
    size_t total_size = 1; /* batch_size */
    /* For the input dimensions [] */
    for (j = 0; j < onnx_state->input_tensors.num_dim[i]; ++j)
    {
      /* [-1, feature numbers] */
      if (onnx_state->input_tensors.dim_values[i][j] <= 0)
      {
        /* Only when input_num_dim[i][0] == -1 */
        if (j != 0 || onnx_state->input_tensors.num_dim[i] == 1)
        {
          onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
          status->error = kAlquimiaErrorEngineIntegrity;
          snprintf(status->message, kAlquimiaMaxStringLength,
                   "ONNX input tensor '%s' has an unsupported dynamic extent.",
                   onnx_state->input_tensors.names[i]);
          CleanupOnSetupFailure(&onnx_state);
          return;
        }
        onnx_state->input_tensors.dim_values[i][j] = 1;
      }
      /* Guard
      ** Check if feature numbers * total size > size_t
      ** In this case, feature numbers * batch size > size_t
      */
      if ((size_t)onnx_state->input_tensors.dim_values[i][j] >
          SIZE_MAX / total_size)
      {
        onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength,
                 "ONNX input tensor '%s' element count overflows size_t.",
                 onnx_state->input_tensors.names[i]);
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
      total_size *= (size_t)onnx_state->input_tensors.dim_values[i][j];
    }
    onnx_state->input_tensors.total_size[i] = total_size;

    /* allocate memory for the input data */
    onnx_state->input_tensors.data[i] =
        (double *)calloc(total_size, sizeof(double));
    if (onnx_state->input_tensors.data[i] == NULL)
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for input_data.");
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    /* The OrtValue wraps input_data; ONNX Runtime does not own that buffer. */
    ort_status = onnx_state->ort.g_ort->CreateTensorWithDataAsOrtValue(
        onnx_state->ort.memory_info, onnx_state->input_tensors.data[i],
        total_size * sizeof(double), onnx_state->input_tensors.dim_values[i],
        onnx_state->input_tensors.num_dim[i],
        ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        &onnx_state->input_tensors.tensor[i]);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Release the type info */
    onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
  }

  onnx_state->output_tensors.names =
      (char **)calloc(num_outputs, sizeof(char *));
  onnx_state->output_tensors.num_dim =
      (size_t *)calloc(num_outputs, sizeof(size_t));
  onnx_state->output_tensors.dim_values =
      (int64_t **)calloc(num_outputs, sizeof(int64_t *));
  onnx_state->output_tensors.total_size =
      (size_t *)calloc(num_outputs, sizeof(size_t));
  onnx_state->output_tensors.data =
      (double **)calloc(num_outputs, sizeof(double *));
  onnx_state->output_tensors.tensor =
      (OrtValue **)calloc(num_outputs, sizeof(OrtValue *));

  if (onnx_state->output_tensors.names == NULL ||
      onnx_state->output_tensors.num_dim == NULL ||
      onnx_state->output_tensors.dim_values == NULL ||
      onnx_state->output_tensors.total_size == NULL ||
      onnx_state->output_tensors.data == NULL ||
      onnx_state->output_tensors.tensor == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for output container arrays.");
    CleanupOnSetupFailure(&onnx_state);
    return;
  }

  /* Parse output names and dimensions */
  for (i = 0; i < num_outputs; ++i)
  {
    char *name = NULL;
    /* Get the names of the output tensor[i] */
    ort_status = onnx_state->ort.g_ort->SessionGetOutputName(
        onnx_state->ort.session, i, onnx_state->ort.allocator, &name);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    onnx_state->output_tensors.names[i] = name;

    OrtTypeInfo *ort_type_info = NULL;  /* Released by ReleaseTypeInfo */
    ort_status = onnx_state->ort.g_ort->SessionGetOutputTypeInfo(
        onnx_state->ort.session, i, &ort_type_info);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    const OrtTensorTypeAndShapeInfo *tensor_info = NULL;  /* Do not free this value. It will be valid until ort_type_info is freed */
    /* Check if the output data type is double */
    ONNXTensorElementDataType onnx_element_type; /* No need to free. enum */
    ort_status = onnx_state->ort.g_ort->CastTypeInfoToTensorInfo(
        ort_type_info, &tensor_info);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Get the output data type */
    ort_status = onnx_state->ort.g_ort->GetTensorElementType(
        tensor_info, &onnx_element_type);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Check if the output data type is double (important for geoscience) */
    if (onnx_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "ONNX output tensor '%s' must have double elements.",
               onnx_state->output_tensors.names[i]);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Record the dimension number */
    size_t dim_count = 0;
    /* Get the number of the dimensions */
    ort_status = onnx_state->ort.g_ort->GetDimensionsCount(
        tensor_info, &dim_count);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    /* For a scalar:
    **
    ** - num_dim = 0
    ** - dim_values = NULL
    ** - total_size = 1
    ** - data[i] holds one double
    ** - OrtValue *tensor[i] is a valid rank-0 tensor

    ** Before OrtRun, routing writes the scalar to data[i][0]. The OrtValue already
    ** references that buffer, so inference reads the assigned value directly. Only
    ** the shape pointer is NULL, indicating rank 0. 
    */
    /* Preserve the model rank. A scalar has zero dimensions and keeps the
    ** NULL dimension array initialized by calloc. */
    onnx_state->output_tensors.num_dim[i] = dim_count;
    if (dim_count > 0)
    {
      onnx_state->output_tensors.dim_values[i] =
          (int64_t *)calloc(dim_count, sizeof(int64_t));
      if (onnx_state->output_tensors.dim_values[i] == NULL)
      {
        onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for output_dim_values.");
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
      /* Get the dimensions */
      ort_status = onnx_state->ort.g_ort->GetDimensions(
          tensor_info, onnx_state->output_tensors.dim_values[i], dim_count);
      if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
      {
        onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
    }

    /* Match the single-state batch convention used for outputs. */
    /* Calculate the total output numbers for each tensor */
    size_t total_size = 1; /* batch_size */
    /* For the output dimensions [] */
    for (j = 0; j < onnx_state->output_tensors.num_dim[i]; ++j)
    {
      /* [-1, feature numbers] */
      if (onnx_state->output_tensors.dim_values[i][j] <= 0)
      {
        /* Only when output_num_dim[i][0] == -1 */
        if (j != 0 || onnx_state->output_tensors.num_dim[i] == 1)
        {
          onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
          status->error = kAlquimiaErrorEngineIntegrity;
          snprintf(status->message, kAlquimiaMaxStringLength,
                   "ONNX output tensor '%s' has an unsupported dynamic extent.",
                   onnx_state->output_tensors.names[i]);
          CleanupOnSetupFailure(&onnx_state);
          return;
        }
        onnx_state->output_tensors.dim_values[i][j] = 1;
      }
      /* Guard
      ** Check if feature numbers * total size > size_t
      ** In this case, feature numbers * batch size > size_t
      */
      if ((size_t)onnx_state->output_tensors.dim_values[i][j] >
          SIZE_MAX / total_size)
      {
        onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength,
                 "ONNX output tensor '%s' element count overflows size_t.",
                 onnx_state->output_tensors.names[i]);
        CleanupOnSetupFailure(&onnx_state);
        return;
      }
      total_size *= (size_t)onnx_state->output_tensors.dim_values[i][j];
    }
    onnx_state->output_tensors.total_size[i] = total_size;

    /* allocate memory for the output data */
    onnx_state->output_tensors.data[i] =
        (double *)calloc(total_size, sizeof(double));
    if (onnx_state->output_tensors.data[i] == NULL)
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength, "Memory allocation failed for output_data.");
      CleanupOnSetupFailure(&onnx_state);
      return;
    }

    /* The OrtValue wraps output_data; ONNX Runtime does not own that buffer. */
    ort_status = onnx_state->ort.g_ort->CreateTensorWithDataAsOrtValue(
        onnx_state->ort.memory_info, onnx_state->output_tensors.data[i],
        total_size * sizeof(double),
        onnx_state->output_tensors.dim_values[i],
        onnx_state->output_tensors.num_dim[i],
        ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
        &onnx_state->output_tensors.tensor[i]);
    if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
    {
      onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    /* Release the type info */
    onnx_state->ort.g_ort->ReleaseTypeInfo(ort_type_info);
  }

  /* Every flattened tensor value must have a complete, valid mapping. */
  for (i = 0; i < onnx_state->input_tensors.num_tensors; ++i)
  {
    /* Guard
    ** avoid total feature inputs > size_t
    */
    if (onnx_state->input_tensors.total_size[i] >
        SIZE_MAX - onnx_state->total_flat_inputs)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Flattened ONNX input element count overflows size_t.");
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    onnx_state->total_flat_inputs += onnx_state->input_tensors.total_size[i];
  }

  for (i = 0; i < onnx_state->output_tensors.num_tensors; ++i)
  {
    /* Guard
    ** avoid total feature outputs > size_t
    */
    if (onnx_state->output_tensors.total_size[i] >
        SIZE_MAX - onnx_state->total_flat_outputs)
    {
      status->error = kAlquimiaErrorEngineIntegrity;
      snprintf(status->message, kAlquimiaMaxStringLength,
               "Flattened ONNX output element count overflows size_t.");
      CleanupOnSetupFailure(&onnx_state);
      return;
    }
    onnx_state->total_flat_outputs +=
        onnx_state->output_tensors.total_size[i];
  }

  /* Set up the mapping rules inside the OnnxEngine->mapping struct 
  ** Initialize AlquimiaSize
  */
  if (!BuildConfigMappings(onnx_state, sizes, status) ||
      // Check if the paired outputs for mobile and immobile exist
      !BuildPairedOutputLookup(onnx_state, status))
  {
    CleanupOnSetupFailure(&onnx_state);
    return;
  }

  /* Each reaction step mutates the reusable tensors and backing buffers in
  ** OnnxEngineState, so shared-state calls are not thread-safe. MPI-only use
  ** remains safe because each process owns its engine state. */
  functionality->thread_safe = false;
  functionality->temperature_dependent =
      HasMappedInput(onnx_state, kFieldTemperature, 0);
  functionality->pressure_dependent =
      HasMappedInput(onnx_state, kFieldAqueousPressure, 0);
  functionality->porosity_update =
      HasMappedOutput(onnx_state, kFieldPorosity, 0);
  functionality->operator_splitting = true;
  functionality->global_implicit = false;
  functionality->index_base = 0;

  *(OnnxEngineState **)onnx_engine_state = onnx_state;
}

/**
 * @brief Releases all adapter and ONNX Runtime resources.
 * @param[in,out] onnx_engine_state Address of the engine pointer created by setup.
 * @param[out] status Returns an invalid-engine error for a NULL engine.
 *
 * Tensor names must be released with the ONNX allocator, while dimensions,
 * buffers, mappings, and copied feature names use the C allocator. On success,
 * the caller's engine pointer is set to NULL.
 */
void onnx_alquimia_shutdown(
    void *onnx_engine_state,
    AlquimiaEngineStatus *status)
{
  /* Input and output tensors are created via independent ORT API calls during setup.
  ** Since there is no strong coupling between them, they are released independently. */
  OnnxEngineState *onnx_state;

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  /* 
  ** Memory Lifecycle & Cleanup Rules:
  ** 1. Input/output tensors and underlying data arrays are created via 
  **    independent API calls during setup and can be freed in any order.
  ** 2. Core ONNX Runtime objects (session, env, etc.) maintain strict 
  **    dependencies and must be released in reverse order of initialization (LIFO).
  */
  if (onnx_engine_state == NULL || *(OnnxEngineState **)onnx_engine_state == NULL)
  {
    status->error = kAlquimiaErrorInvalidEngine;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid ONNX engine state pointer in shutdown.");
    return;
  }

  onnx_state = *(OnnxEngineState **)onnx_engine_state;
  if (onnx_state->ort.g_ort != NULL)
  {
    /* Mapping feature names point into onnx_config (freed below). */
    free(onnx_state->input_mappings);
    onnx_state->input_mappings = NULL;
    free(onnx_state->output_mappings);
    onnx_state->output_mappings = NULL;
    free(onnx_state->output_has_paired_mapping);
    onnx_state->output_has_paired_mapping = NULL;

    /* Input and output collections own independent tensors and buffers. */
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

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  (void)aux_data;
  (void)properties;

  if (onnx_engine_state == NULL || *(OnnxEngineState **)onnx_engine_state == NULL)
  {
    status->error = kAlquimiaErrorInvalidEngine;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid ONNX engine state pointer in ProcessCondition.");
    return;
  }

  onnx_state = *(OnnxEngineState **)onnx_engine_state;

  if (state == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid state pointer in ProcessCondition.");
    return;
  }

  // Initial condition provided by JSON
  if (onnx_state->hands_off)
  {
    // Point to the related onnx_config.conditions
    const OnnxAlquimiaCondition *matching_condition = NULL;
    size_t condition_index;
    
    // Passed by the driver
    if (condition != NULL && condition->name != NULL)
    {
      for (condition_index = 0;
           condition_index < onnx_state->onnx_config.num_conditions;
           ++condition_index)
      {
        const OnnxAlquimiaCondition *current_condition =
            &onnx_state->onnx_config.conditions[condition_index];
        // Find the related initial condition in the JSON file
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

  // Initial condition provided by driver
  /* An absent condition intentionally preserves all existing state values. */
  if (condition == NULL || condition->aqueous_constraints.data == NULL || condition->aqueous_constraints.size <= 0)
  {
    return;
  }

  /* Generic driver constraints have no explicit field identity. Validate all
   * destinations before writing, so ambiguous names cannot update both phases.
   *
   * TODO / PLACEHOLDER: Waiting for upstream driver logic to mature.
   * Currently, the driver's AlquimiaAqueousConstraint data structure is incomplete 
   * for this use case: it provides a component name and a value, but lacks explicit 
   * physical state labels (e.g., specifying if it belongs to mobile or immobile phases).
   * 
   * TWO-PASS TRANSACTIONAL UPDATE (All-or-Nothing):
   * Because the incoming driver data is ambiguous, we must use a two-pass approach 
   * to prevent partial memory corruption (which would violate mass conservation):
   * 
   * - Pass 0 (Dry-Run / Validation): Iterates through all constraints and validates 
   *   them against the engine mappings WITHOUT writing to memory. If an ambiguous 
   *   name is detected (e.g., trying to blindly update both aqueous and solid phases), 
   *   the function safely aborts leaving the existing state untouched.
   * - Pass 1 (Execution): Once Pass 0 confirms all constraints map 1-to-1 safely, 
   *   this pass formally writes the values into the Alquimia state memory slots.
   * 
   * Note: Once the driver interface is updated to pass explicit state identities, 
   * this defensive two-pass logic can be refactored into a standard one-pass assignment.
   */
  for (int pass = 0; pass < 2; ++pass)
  {
    for (k = 0; k < onnx_state->total_flat_inputs; ++k)
    {
      const AlquimiaAqueousConstraint *constraint = NULL;
      for (int c_idx = 0; c_idx < condition->aqueous_constraints.size; ++c_idx)
      {
        if (strcmp(condition->aqueous_constraints.data[c_idx].primary_species_name,
                   onnx_state->input_mappings[k].feature) == 0)
        {
          constraint = &condition->aqueous_constraints.data[c_idx];
          break;
        }
      }
      if (constraint == NULL)
      {
        continue;
      }
      OnnxAlquimiaConditionItem item = {0};
      FeatureMapping mapping;

      /* Treat as scalar shorthand because the driver lacks explicit state info */
      item.feature = constraint->primary_species_name;
      item.value = constraint->value;
      if (!ResolveConditionMapping(onnx_state, &item, &mapping, status))
      {
        return;
      }

      /* Pass 1 formally writes the data */
      if (pass == 1 && mapping.feature != NULL)
      {
        SetAlquimiaValue(state, mapping, item.value, status);
        if (status->error != kAlquimiaNoError)
        {
          return;
        }
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
  int i;
  double old_porosity;
  double new_porosity;

  status->error = kAlquimiaNoError;
  status->message[0] = '\0';

  /* Input/output tensors will be reused
  ** They will be freed in the shutdown*/
  // Unused
  (void)delta_t;
  (void)aux_data;
  (void)natural_id;

  if (onnx_engine_state == NULL || *(OnnxEngineState **)onnx_engine_state == NULL)
  {
    status->error = kAlquimiaErrorInvalidEngine;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid ONNX engine state pointer in reactionstepoperatorsplit.");
    return;
  }

  onnx_state = *(OnnxEngineState **)onnx_engine_state;

  if (state == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid state in reactionstepoperatorsplit.");
    return;
  }

  old_porosity = state->porosity;
  new_porosity = old_porosity;
  {
    size_t flat_index = 0;
    for (i = 0; i < (int)onnx_state->input_tensors.num_tensors; ++i)
    {
      size_t k;
      for (k = 0; k < onnx_state->input_tensors.total_size[i]; ++k)
      {
        onnx_state->input_tensors.data[i][k] = GetAlquimiaModelInput(
            state, properties, onnx_state->input_mappings[flat_index], status);
        if (status->error != kAlquimiaNoError)
        {
          return;
        }
        flat_index++;
      }
    }
  }

  /* Run inference using preallocated input and output tensors and dynamic names */
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

  /* Use the final porosity for closure regardless of output mapping order.
  ** Output tensors wrap the adapter's preallocated data arrays. */
  {
    size_t flat_index = 0;
    for (i = 0; i < (int)onnx_state->output_tensors.num_tensors; ++i)
    {
      size_t k;
      for (k = 0; k < onnx_state->output_tensors.total_size[i]; ++k)
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
        ++flat_index;
      }
    }
  }

  /* Copy output data back through the required explicit mappings. */
  {
    size_t flat_index = 0;
    for (i = 0; i < (int)onnx_state->output_tensors.num_tensors; ++i)
    {
      /* Temporary output array used to store the data from the tensor */
      double *out_arr = NULL;
      /* Extract the data from the output tensor */
      ort_status = onnx_state->ort.g_ort->GetTensorMutableData(
          onnx_state->output_tensors.tensor[i], (void **)&out_arr);
      if (!CheckStatus(onnx_state->ort.g_ort, ort_status, status))
      {
        return;
      }

      if (out_arr == NULL)
      {
        status->error = kAlquimiaErrorEngineIntegrity;
        snprintf(status->message, kAlquimiaMaxStringLength, "GetTensorMutableData returned NULL output buffer for output tensor %d.", i);
        return;
      }

      size_t k;
      for (k = 0; k < onnx_state->output_tensors.total_size[i]; ++k)
      {
        SetAlquimiaModelOutput(
            state, properties, old_porosity, new_porosity,
            onnx_state->output_mappings[flat_index],
            onnx_state->output_has_paired_mapping[flat_index], out_arr[k],
            status);
        if (status->error != kAlquimiaNoError)
        {
          return;
        }
        flat_index++;
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

  if (onnx_engine_state == NULL || *(OnnxEngineState **)onnx_engine_state == NULL)
  {
    status->error = kAlquimiaErrorInvalidEngine;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Invalid ONNX engine state pointer.");
    return;
  }

  /* Unused data */
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

  if (onnx_engine_state == NULL || *(OnnxEngineState **)onnx_engine_state == NULL)
  {
    status->error = kAlquimiaErrorInvalidEngine;
    snprintf(status->message, kAlquimiaMaxStringLength, "Invalid ONNX engine state pointer.");
    return;
  }

  onnx_state = *(OnnxEngineState **)onnx_engine_state;
  if (meta_data == NULL)
  {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength,
             "Invalid problem metadata destination for ONNX engine.");
    return;
  }

  for (i = 0; i < onnx_state->total_flat_inputs; ++i)
  {
    /* Directly point to the OnnxEngine-> input_mappings */ 
    FeatureMapping mapping = onnx_state->input_mappings[i];
    AlquimiaVectorString *metadata_name = MetadataNamesForMapping(
        meta_data, mapping.alquimia_state);
    /* All the manipulation is to the OnnxEngine */
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
