/* -*-  mode: c; c-default-style: "google"; indent-tabs-mode: nil -*- */

#include "alquimia/onnx_alquimia_config.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cJSON.h"

/* cJSON is confined to this translation unit. The interface layer receives
** only Alquimia-owned structures whose strings remain valid after the parse
** tree is deleted. */

/* Returns a malloc'd copy of a cJSON-owned string, or NULL. */
static char *CopyString(const char *str_value)
{
  size_t length = strlen(str_value);
  char *copy = (char *)malloc(length + 1);
  if (copy != NULL)
  {
    memcpy(copy, str_value, length + 1);
  }
  return copy;
}

/* True when name exactly matches one of the allowed property names. */
static bool IsAllowedProperty(
    const char *name,
    const char *const *allowed_schema_fields,
    size_t num_allowed_schema_fields)
{
  size_t i;
  for (i = 0; i < num_allowed_schema_fields; ++i)
  {
    if (strcmp(name, allowed_schema_fields[i]) == 0)
    {
      return true;
    }
  }
  return false;
}

/**
 * @brief Rejects bad or duplicate member names in one JSON object.
 * @param[in] cjson_object Object whose direct members are checked.
 * @param[in] allowed Fixed schema property names, or NULL for objects whose
 *        member names are user-defined (conditions, features).
 * @param[in] num_allowed Number of entries in @p allowed.
 * @param[in] context Human-readable object name used in errors.
 * @param[out] error_message Destination for the first validation error.
 * @param[in] error_message_size Size of @p error_message in bytes.
 * @return True when every name is allowed (or nonempty) and unique.
 *
 * cJSON accepts duplicate keys and its lookup returns only one of them, so
 * duplicates are found by scanning the member list.
 */
static bool ValidateMembers(
    const cJSON *cjson_object,
    const char *const *allowed,
    size_t num_allowed,
    const char *context,
    char *error_message,
    size_t error_message_size)
{
  const cJSON *cjson_member;

  cJSON_ArrayForEach(cjson_member, cjson_object)
  {
    const char *name = cjson_member->string;
    const cJSON *cjson_other;

    if (allowed != NULL &&
        (name == NULL || !IsAllowedProperty(name, allowed, num_allowed)))
    {
      snprintf(error_message, error_message_size,
               "Unknown property '%s' in %s.", name == NULL ? "" : name,
               context);
      return false;
    }
    if (allowed == NULL && (name == NULL || name[0] == '\0'))
    {
      snprintf(error_message, error_message_size,
               "Every name in %s must be nonempty.", context);
      return false;
    }
    for (cjson_other = cjson_member->next; cjson_other != NULL;
         cjson_other = cjson_other->next)
    {
      if (cjson_other->string != NULL &&
          strcmp(name, cjson_other->string) == 0)
      {
        snprintf(error_message, error_message_size,
                 allowed != NULL ? "Duplicate property '%s' in %s."
                                 : "Duplicate name '%s' in %s.",
                 name, context);
        return false;
      }
    }
  }
  return true;
}

/**
 * @brief Reads and copies a required nonempty string property.
 * @param[in] cjson_object Object containing the property.
 * @param[in] name Exact property name.
 * @param[in] context Human-readable object name used in errors.
 * @param[out] str_value Receives C-allocator-owned storage on success.
 * @param[out] error_message Destination for validation or allocation errors.
 * @param[in] error_message_size Size of @p error_message in bytes.
 * @return True when the property is a nonempty string and copying succeeds.
 *
 * Copying detaches the config representation from the cJSON tree lifetime.
 */
static bool GetRequiredString(
    const cJSON *cjson_object,
    const char *name,
    const char *context,
    char **str_value,
    char *error_message,
    size_t error_message_size)
{
  const cJSON *cjson_item = cJSON_GetObjectItemCaseSensitive(cjson_object, name);
  if (!cJSON_IsString(cjson_item) || cjson_item->valuestring == NULL ||
      cjson_item->valuestring[0] == '\0')
  {
    snprintf(error_message, error_message_size,
             "Required property '%s' in %s must be a nonempty string.",
             name, context);
    return false;
  }
  *str_value = CopyString(cjson_item->valuestring);
  if (*str_value == NULL)
  {
    snprintf(error_message, error_message_size,
             "Memory allocation failed while loading ONNX config.");
    return false;
  }
  return true;
}

/**
 * @brief Reads a required integer representable as a nonnegative C int.
 * @param[in] cjson_object Object containing the property.
 * @param[in] name Exact property name.
 * @param[in] context Human-readable object name used in errors.
 * @param[out] int_value Receives the parsed integer on success.
 * @param[out] error_message Destination for validation errors.
 * @param[in] error_message_size Size of @p error_message in bytes.
 * @return True when the JSON number is integral and lies in [0, INT_MAX].
 *
 * cJSON stores numbers as double, so the round-trip comparison rejects
 * fractional values before conversion to the config's integer fields.
 */
static bool GetRequiredInteger(
    const cJSON *cjson_object,
    const char *name,
    const char *context,
    int *int_value,
    char *error_message,
    size_t error_message_size)
{
  const cJSON *cjson_item = cJSON_GetObjectItemCaseSensitive(cjson_object, name);
  if (!cJSON_IsNumber(cjson_item) || cjson_item->valuedouble < 0.0 ||
      cjson_item->valuedouble > INT_MAX ||
      (double)(int)cjson_item->valuedouble != cjson_item->valuedouble)
  {
    snprintf(error_message, error_message_size,
             "Required property '%s' in %s must be a nonnegative integer.",
             name, context);
    return false;
  }
  *int_value = (int)cjson_item->valuedouble;
  return true;
}

/**
 * @brief Resolves the config model entry to an owned filesystem path.
 * @param[in] config_path Path used to open the config.
 * @param[in] model Nonempty model entry from the config.
 * @param[out] model_path Receives newly allocated path storage.
 * @param[out] error_message Destination for overflow or allocation errors.
 * @param[in] error_message_size Size of @p error_message in bytes.
 * @return True when path construction succeeds.
 *
 * Absolute paths are preserved. Relative paths use the config's directory,
 * not the process working directory; existence is checked later by setup.
 */
static bool ResolveModelPath(
    const char *config_path,
    const char *model,
    char **model_path,
    char *error_message,
    size_t error_message_size)
{
  const char *last_separator;
  size_t directory_length;
  size_t model_length = strlen(model);

  if (model[0] == '/')
  {
    *model_path = CopyString(model);
  }
  else
  {
    last_separator = strrchr(config_path, '/');
    directory_length = last_separator == NULL ? 0 : (size_t)(last_separator - config_path) + 1;
    *model_path = (char *)malloc(directory_length + model_length + 1);
    if (*model_path != NULL)
    {
      memcpy(*model_path, config_path, directory_length);
      memcpy(*model_path + directory_length, model, model_length + 1);
    }
  }

  if (*model_path == NULL)
  {
    snprintf(error_message, error_message_size,
             "Memory allocation failed while resolving ONNX model path.");
    return false;
  }
  return true;
}

/**
 * @brief Parses input or output mappings into owned config storage.
 * @param[in] cjson_array Validated JSON array from the config root.
 * @param[in] mapping_role Human-readable mapping role, such as "input".
 * @param[out] mappings Receives an allocated array and owned strings.
 * @param[out] num_mappings Receives the number of parsed mappings.
 * @param[out] error_message Destination for validation or allocation errors.
 * @param[in] error_message_size Size of @p error_message in bytes.
 * @return True when every mapping satisfies the versioned contract.
 *
 * Input and output mapping objects have the same schema. Role-specific
 * behavior, such as condition coverage, is validated outside this parser.
 * On failure, the caller releases partially populated entries through
 * OnnxAlquimiaFreeConfig.
 */
static bool ParseMappings(
    const cJSON *cjson_array,
    const char *mapping_role,
    OnnxAlquimiaMapping **mappings,
    size_t *num_mappings,
    char *error_message,
    size_t error_message_size)
{
  static const char *const allowed_schema_fields[] = {
      "tensor", "tensor_element_index", "feature", "alquimia_state",
      "alquimia_state_index"};
  char mapping_context[32];
  cJSON *cjson_item;
  size_t i = 0;
  int count = cJSON_GetArraySize(cjson_array);

  snprintf(mapping_context, sizeof(mapping_context), "%s mapping", mapping_role);
  *num_mappings = (size_t)count;
  if (count > 0)
  {
    *mappings = (OnnxAlquimiaMapping *)calloc((size_t)count, sizeof(**mappings));
    if (*mappings == NULL)
    {
      snprintf(error_message, error_message_size,
               "Memory allocation failed for ONNX %s mappings.", mapping_role);
      return false;
    }
  }
  cJSON_ArrayForEach(cjson_item, cjson_array)
  {
    int tensor_element_index;
    OnnxAlquimiaMapping *mapping = &(*mappings)[i];
    if (!cJSON_IsObject(cjson_item))
    {
      snprintf(error_message, error_message_size,
               "Every ONNX %s mapping must be an object.", mapping_role);
      return false;
    }
    if (!ValidateMembers(cjson_item, allowed_schema_fields, 5, mapping_context,
                         error_message, error_message_size) ||
        !GetRequiredString(cjson_item, "tensor", mapping_context,
                           &mapping->tensor, error_message, error_message_size) ||
        !GetRequiredInteger(cjson_item, "tensor_element_index", mapping_context,
                            &tensor_element_index, error_message,
                            error_message_size) ||
        !GetRequiredString(cjson_item, "feature", mapping_context,
                           &mapping->feature, error_message, error_message_size) ||
        !GetRequiredString(cjson_item, "alquimia_state", mapping_context,
                           &mapping->alquimia_state, error_message,
                           error_message_size) ||
        !GetRequiredInteger(cjson_item, "alquimia_state_index", mapping_context,
                            &mapping->alquimia_state_index, error_message,
                            error_message_size))
    {
      return false;
    }
    mapping->tensor_element_index = (size_t)tensor_element_index;
    ++i;
  }
  return true;
}

/**
 * @brief Verifies that every condition supplies every mapped input feature.
 * @param[in] onnx_config Config containing parsed inputs and conditions.
 * @param[out] error_message Destination for the first validation error.
 * @param[in] error_message_size Size of @p error_message in bytes.
 * @return True when all mapped features exist in each condition.
 *
 * Conditions may contain additional features so a shared condition can serve
 * models with different input subsets.
 */
static bool ValidateConditionFeature(
    const OnnxAlquimiaConfig *onnx_config,
    char *error_message,
    size_t error_message_size)
{
  size_t condition_index;
  size_t input_index;

  for (condition_index = 0;
       condition_index < onnx_config->num_conditions;
       ++condition_index)
  {
    const OnnxAlquimiaCondition *condition =
        &onnx_config->conditions[condition_index];
    for (input_index = 0; input_index < onnx_config->num_inputs; ++input_index)
    {
      const char *required_feature = onnx_config->inputs[input_index].feature;
      size_t item_index;
      bool found = false;
      for (item_index = 0; item_index < condition->num_items; ++item_index)
      {
        if (strcmp(condition->items[item_index].feature,
                   required_feature) == 0)
        {
          found = true;
          break;
        }
      }
      if (!found)
      {
        snprintf(error_message, error_message_size,
                 "Condition '%s' is missing input feature '%s'.",
                 condition->name, required_feature);
        return false;
      }
    }
  }
  return true;
}

/**
 * @brief Parses optional named JSON initial conditions.
 * @param[in] cjson_conditions Conditions object from the config root.
 * @param[in,out] onnx_config Destination whose condition storage becomes owned.
 * @param[out] error_message Destination for validation or allocation errors.
 * @param[in] error_message_size Size of @p error_message in bytes.
 * @return True when all names and numeric feature values are valid.
 */
static bool ParseConditions(
    const cJSON *cjson_conditions,
    OnnxAlquimiaConfig *onnx_config,
    char *error_message,
    size_t error_message_size)
{
  const cJSON *cjson_condition;
  size_t condition_index = 0;
  int condition_count;

  if (cjson_conditions == NULL)
  {
    return true;
  }
  
  /* "conditions": {"<name>": {"<feature>": <number>, or
  **                            "<feature>": {"<state field>": <number>, ...}}} */
  if (!cJSON_IsObject(cjson_conditions))
  {
    snprintf(error_message, error_message_size,
             "ONNX config conditions must be an object.");
    return false;
  }
  if (!ValidateMembers(cjson_conditions, NULL, 0, "conditions",
                            error_message, error_message_size))
  {
    return false;
  } 

  condition_count = cJSON_GetArraySize(cjson_conditions);

  onnx_config->num_conditions = (size_t)condition_count;
  if (condition_count > 0)
  {
    onnx_config->conditions = (OnnxAlquimiaCondition *)calloc(
        (size_t)condition_count, sizeof(*onnx_config->conditions));
    if (onnx_config->conditions == NULL)
    {
      snprintf(error_message, error_message_size,
               "Memory allocation failed for ONNX conditions.");
      return false;
    }
  }

  cJSON_ArrayForEach(cjson_condition, cjson_conditions)
  {
    OnnxAlquimiaCondition *condition =
        &onnx_config->conditions[condition_index];
    const cJSON *cjson_item;
    size_t item_index = 0;
    size_t item_count = 0;

    condition->name = CopyString(cjson_condition->string);
    if (condition->name == NULL)
    {
      snprintf(error_message, error_message_size,
               "Memory allocation failed for an ONNX condition name.");
      return false;
    }

    if (!cJSON_IsObject(cjson_condition))
    {
      snprintf(error_message, error_message_size,
               "Condition '%s' must be an object.", condition->name);
      return false;
    }

    if (!ValidateMembers(cjson_condition, NULL, 0, "condition feature values",
                              error_message, error_message_size))
    {
      return false;
    }

    /* Reserve one assignment per explicit field, or per scalar. */
    cJSON_ArrayForEach(cjson_item, cjson_condition)
    {
      size_t count = 1;
      if (cJSON_IsObject(cjson_item))
      {
        if (!ValidateMembers(cjson_item, NULL, 0, "condition fields",
                                  error_message, error_message_size))
        {
          return false;
        }
        count = (size_t)cJSON_GetArraySize(cjson_item);
        if (count == 0)
        {
          snprintf(error_message, error_message_size,
                   "Feature '%s' in condition '%s' has no fields.",
                   cjson_item->string, condition->name);
          return false;
        }
      }
      item_count += count;
    }
    condition->num_items = item_count;
    if (item_count > 0)
    {
      condition->items = (OnnxAlquimiaConditionItem *)calloc(
          (size_t)item_count, sizeof(*condition->items));
      if (condition->items == NULL)
      {
        snprintf(error_message, error_message_size,
                 "Memory allocation failed for ONNX condition values.");
        return false;
      }
    }
    cJSON_ArrayForEach(cjson_item, cjson_condition)
    {
      /* Nested form names the state field(s) explicitly, e.g.
      ** "Zn": {"total_mobile": 80, "total_immobile": 70}. */
      bool nested = cJSON_IsObject(cjson_item);
      const cJSON *value = nested ? cjson_item->child : cjson_item;
      do
      {
        OnnxAlquimiaConditionItem *item = &condition->items[item_index];
        if (!cJSON_IsNumber(value) || !isfinite(value->valuedouble))
        {
          snprintf(error_message, error_message_size,
                   "Feature '%s' in condition '%s' must be a finite number.",
                   cjson_item->string, condition->name);
          return false;
        }
        item->feature = CopyString(cjson_item->string);
        if (nested)
        {
          item->alquimia_state = CopyString(value->string);
        }
        if (item->feature == NULL || (nested && item->alquimia_state == NULL))
        {
          snprintf(error_message, error_message_size,
                   "Memory allocation failed for an ONNX condition field.");
          return false;
        }
        item->value = value->valuedouble;
        ++item_index;
        value = nested ? value->next : NULL;
      } while (value != NULL);
    }
    ++condition_index;
  }

  return ValidateConditionFeature(
      onnx_config, error_message, error_message_size);
}

/**
 * @brief Reads a complete config file into a null-terminated buffer.
 * @param[in] config_path config filesystem path.
 * @param[out] json_contents Returns newly allocated file json_contents.
 * @param[out] file_length Receives the file length, excluding the terminator.
 * @param[out] error_message Destination for read or allocation errors.
 * @param[in] error_message_size Size of @p error_message in bytes.
 * @return True on success; false after closing the file and freeing partial
 *         storage.
 *
 * This helper owns the FILE for its entire lifetime, keeping file cleanup out
 * of the parser's validation control flow.
 */
static bool ReadConfigJsonContents(
    const char *config_path,
    char **json_contents,
    size_t *file_length,
    char *error_message,
    size_t error_message_size)
{
  FILE *file;
  long file_size;
  size_t bytes_read;

  *json_contents = NULL;
  *file_length = 0;
  file = fopen(config_path, "rb");
  /* Size the buffer by seeking to the end. */
  if (file == NULL || fseek(file, 0, SEEK_END) != 0 ||
      (file_size = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0)
  {
    if (file != NULL)
    {
      fclose(file);
    }
    snprintf(error_message, error_message_size,
             "Unable to read ONNX config: %s", config_path);
    return false;
  }

  *json_contents = (char *)malloc((size_t)file_size + 1);
  if (*json_contents == NULL)
  {
    fclose(file);
    snprintf(error_message, error_message_size,
             "Memory allocation failed while reading ONNX config.");
    return false;
  }
  bytes_read = fread(*json_contents, 1, (size_t)file_size, file);
  fclose(file);
  if (bytes_read != (size_t)file_size)
  {
    free(*json_contents);
    *json_contents = NULL;
    snprintf(error_message, error_message_size,
             "Failed to read the complete ONNX config.");
    return false;
  }

  (*json_contents)[bytes_read] = '\0';
  *file_length = bytes_read;
  return true;
}

/**
 * @brief Validates the root object and populates owned config storage.
 * @param[in] config_path JSON config path used to resolve a relative model entry.
 * @param[in] cjson_root Parsed cJSON root; ownership remains with the caller.
 * @param[in,out] onnx_config Destination for copied paths and mapping specifications.
 * @param[out] error_message Destination for schema or allocation errors.
 * @param[in] error_message_size Size of @p error_message in bytes.
 * @return True when the root satisfies the complete version contract.
 */
static bool PopulateConfig(
    const char *config_path,
    const cJSON *cjson_root,
    OnnxAlquimiaConfig *onnx_config,
    char *error_message,
    size_t error_message_size)
{
  static const char *const allowed_schema_fields[] = {
      "schema_version", "model", "conditions", "inputs", "outputs"};
  const cJSON *cjson_schema_version;
  const cJSON *cjson_model;
  const cJSON *cjson_conditions;
  const cJSON *cjson_inputs;
  const cJSON *cjson_outputs;
  if (!cJSON_IsObject(cjson_root))
  {
    snprintf(error_message, error_message_size,
             "ONNX config root must be an object.");
    return false;
  }
  if (!ValidateMembers(cjson_root, allowed_schema_fields, 5, "config root",
                          error_message, error_message_size))
  {
    return false;
  }

  cjson_schema_version = cJSON_GetObjectItemCaseSensitive(cjson_root, "schema_version");
  cjson_model = cJSON_GetObjectItemCaseSensitive(cjson_root, "model");
  cjson_conditions = cJSON_GetObjectItemCaseSensitive(cjson_root, "conditions");
  cjson_inputs = cJSON_GetObjectItemCaseSensitive(cjson_root, "inputs");
  cjson_outputs = cJSON_GetObjectItemCaseSensitive(cjson_root, "outputs");
  if (!cJSON_IsNumber(cjson_schema_version) || cjson_schema_version->valuedouble != 1.0)
  {
    snprintf(error_message, error_message_size,
             "ONNX config schema_version must be integer 1.");
    return false;
  }
  if (!cJSON_IsString(cjson_model) || cjson_model->valuestring == NULL ||
      cjson_model->valuestring[0] == '\0')
  {
    snprintf(error_message, error_message_size,
             "ONNX config model must be a nonempty string.");
    return false;
  }
  if (!cJSON_IsArray(cjson_inputs) || !cJSON_IsArray(cjson_outputs))
  {
    snprintf(error_message, error_message_size,
             "ONNX config inputs and outputs must be arrays.");
    return false;
  }
  /* Input and output arrays share one mapping schema. Conditions are parsed
  ** afterward because their coverage applies only to model inputs. */
  return ResolveModelPath(config_path, cjson_model->valuestring, &onnx_config->model_path,
                          error_message, error_message_size) &&
         ParseMappings(cjson_inputs, "input", &onnx_config->inputs,
                       &onnx_config->num_inputs, error_message,
                       error_message_size) &&
         ParseMappings(cjson_outputs, "output", &onnx_config->outputs,
                       &onnx_config->num_outputs, error_message,
                       error_message_size) &&
         ParseConditions(cjson_conditions, onnx_config, error_message,
                         error_message_size);
}

/* Frees a mapping array (may be NULL) and its strings. */
static void FreeMappings(
    OnnxAlquimiaMapping *mappings,
    size_t num_mappings)
{
  size_t i;

  for (i = 0; i < num_mappings; ++i)
  {
    free(mappings[i].tensor);
    free(mappings[i].feature);
    free(mappings[i].alquimia_state);
  }
  free(mappings);
}

/**
 * @brief Releases a complete or partially initialized config.
 * @param[in,out] onnx_config config whose owned paths, mappings, and strings are freed.
 *
 * The structure is zeroed after cleanup, making repeated cleanup and setup
 * failure paths safe.
 */
void OnnxAlquimiaFreeConfig(OnnxAlquimiaConfig *onnx_config)
{
  size_t i;
  if (onnx_config == NULL)
  {
    return;
  }
  free(onnx_config->model_path);
  for (i = 0; onnx_config->conditions != NULL &&
              i < onnx_config->num_conditions; ++i)
  {
    size_t j;
    free(onnx_config->conditions[i].name);
    for (j = 0; onnx_config->conditions[i].items != NULL &&
                j < onnx_config->conditions[i].num_items; ++j)
    {
      free(onnx_config->conditions[i].items[j].feature);
      free(onnx_config->conditions[i].items[j].alquimia_state);
    }
    free(onnx_config->conditions[i].items);
  }
  free(onnx_config->conditions);
  FreeMappings(onnx_config->inputs, onnx_config->num_inputs);
  FreeMappings(onnx_config->outputs, onnx_config->num_outputs);
  memset(onnx_config, 0, sizeof(*onnx_config));
}

/**
 * @brief Loads and strictly validates a versioned ONNX sidecar config.
 * @param[in] config_path JSON filesystem path.
 * @param[in,out] onnx_config Returns owned model-path and mapping storage on success.
 * @param[out] error_message Destination for the first read, parse, or schema error.
 * @param[in] error_message_size Size of @p error_message in bytes.
 * @return True on success; false after releasing all partial allocations.
 *
 * Parsing is length-aware, requires one complete JSON document, and rejects
 * unknown or duplicate properties. No cJSON-owned pointer escapes this call.
 */
bool OnnxAlquimiaLoadConfig(
    const char *config_path,
    OnnxAlquimiaConfig *onnx_config,
    char *error_message,
    size_t error_message_size)
{
  char *json_contents = NULL;
  size_t bytes_read;
  const char *parse_end = NULL;
  cJSON *cjson_root = NULL;
  bool success;

  memset(onnx_config, 0, sizeof(*onnx_config));
  if (config_path == NULL || config_path[0] == '\0')
  {
    snprintf(error_message, error_message_size,
             "ONNX config file path not provided.");
    return false;
  }
  if (!ReadConfigJsonContents(config_path, &json_contents, &bytes_read,
                              error_message, error_message_size))
  {
    return false;
  }

  /* Requiring the terminator and checking parse_end rejects otherwise-valid
  ** JSON followed by non-whitespace trailing content. */
  cjson_root = cJSON_ParseWithLengthOpts(
      json_contents, bytes_read + 1, &parse_end, true);
  if (cjson_root == NULL || parse_end == NULL ||
      (size_t)(parse_end - json_contents) != bytes_read)
  {
    snprintf(error_message, error_message_size,
             "ONNX config is not valid strict JSON.");
    cJSON_Delete(cjson_root);
    free(json_contents);
    return false;
  }

  success = PopulateConfig(config_path, cjson_root, onnx_config,
                             error_message, error_message_size);
  cJSON_Delete(cjson_root);
  free(json_contents);
  if (!success)
  {
    OnnxAlquimiaFreeConfig(onnx_config);
  }
  return success;
}
