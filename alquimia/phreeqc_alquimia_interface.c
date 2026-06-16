/* -*-  mode: c++; c-default-style: "google"; indent-tabs-mode: nil -*- */

#include "alquimia/phreeqc_alquimia_interface.h"
#include "alquimia/alquimia_memory.h"
#include "alquimia/alquimia_util.h"
#include "alquimia/alquimia_constants.h"
#include "RM_interface_C.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void phreeqc_alquimia_setup(const char* input_filename,
                            bool hands_off,
                            void* pft_engine_state,
                            AlquimiaSizes* sizes,
                            AlquimiaEngineFunctionality* functionality,
                            AlquimiaEngineStatus* status) {
  
  if (pft_engine_state == NULL) {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "phreeqc_alquimia_setup: engine_state pointer is NULL.");
    return;
  }

  PhreeqcRMEngineState* engine = (PhreeqcRMEngineState*)malloc(sizeof(PhreeqcRMEngineState));
  if (!engine) {
    status->error = 1;
    snprintf(status->message, kAlquimiaMaxStringLength, "phreeqc_alquimia_setup: Memory allocation failed for engine state.");
    return;
  }

  /* Initialize a single PhreeqcRM instance (nxyz=1, nthreads=1) */
  int nxyz = 1;
  int nthreads = 1;
#if defined(USE_MPI) && 0
  /* Alquimia builds with MPI, but PhreeqcRM can be built sequentially.
     If we configured PhreeqcRM with MPI=OFF, it exposes RM_Create(nxyz, nthreads)
     instead of RM_Create(nxyz, comm). Since we built it statically with MPI=OFF
     in Superbuild, the signature is RM_Create(int, int). */
#endif
  engine->rm_id = RM_Create(nxyz, nthreads);
  if (engine->rm_id < 0) {
    status->error = 1;
    snprintf(status->message, kAlquimiaMaxStringLength, "phreeqc_alquimia_setup: Failed to create PhreeqcRM instance.");
    free(engine);
    return;
  }

  /* Set unit scaling properties */
  /* Mobile: mol / L solution */
  RM_SetUnitsSolution(engine->rm_id, 2); 
  
  /* Immobile: mol / L of Representative Volume */
  RM_SetUnitsPPassemblage(engine->rm_id, 0); 
  RM_SetUnitsExchange(engine->rm_id, 0);
  RM_SetUnitsSurface(engine->rm_id, 0);
  RM_SetUnitsGasPhase(engine->rm_id, 0);
  RM_SetUnitsKinetics(engine->rm_id, 0);
  RM_SetUnitsSSassemblage(engine->rm_id, 0);

  /* Force PhreeqcRM to strictly use our provided density and volume for conversions */
  RM_UseSolutionDensityVolume(engine->rm_id, 0);
  
  /* Set Representative Volume to 1 Liter so 1 L = 0.001 m^3_bulk mapping works perfectly */
  double rv = 1.0;
  RM_SetRepresentativeVolume(engine->rm_id, &rv);

  /* We want to access species concentrations */
  RM_SetSpeciesSaveOn(engine->rm_id, 1);

  /* Run the input file to define databases and initial conditions */
  int rm_status = RM_RunFile(engine->rm_id, 1, 1, 1, input_filename);
  if (rm_status < 0) {
    status->error = rm_status;
    snprintf(status->message, kAlquimiaMaxStringLength, "phreeqc_alquimia_setup: RM_RunFile failed for %s.", input_filename);
    RM_Destroy(engine->rm_id);
    free(engine);
    return;
  }

  /* Analyze the components */
  engine->num_primary = RM_FindComponents(engine->rm_id);
  if (engine->num_primary < 0) {
    status->error = engine->num_primary;
    snprintf(status->message, kAlquimiaMaxStringLength, "phreeqc_alquimia_setup: RM_FindComponents failed.");
    RM_Destroy(engine->rm_id);
    free(engine);
    return;
  }

  /* Inject a SELECTED_OUTPUT block to retrieve pH and mineral saturation indices later */
  char sel_out_string[1024];
  snprintf(sel_out_string, 1024, "SELECTED_OUTPUT 333\n-pH\n-saturation_indices\n");
  RM_RunString(engine->rm_id, 1, 0, 0, sel_out_string);
  RM_SetCurrentSelectedOutputUserNumber(engine->rm_id, 333);
  RM_SetSelectedOutputOn(engine->rm_id, 1);

  /* Cache sizes */
  int total_species = RM_GetSpeciesCount(engine->rm_id);
  engine->num_aqueous_complexes = total_species - engine->num_primary;
  engine->num_minerals = RM_GetEquilibriumPhasesCount(engine->rm_id);
  engine->num_surface_sites = RM_GetSurfaceSpeciesCount(engine->rm_id);
  engine->num_ion_exchange_sites = RM_GetExchangeSpeciesCount(engine->rm_id);
  engine->num_gases = RM_GetGasComponentsCount(engine->rm_id);
  engine->num_sel_out_cols = RM_GetSelectedOutputColumnCount(engine->rm_id);

  /* Pass cached sizes to Alquimia API */
  sizes->num_primary = engine->num_primary;
  sizes->num_aqueous_complexes = engine->num_aqueous_complexes;
  sizes->num_minerals = engine->num_minerals;
  sizes->num_surface_sites = engine->num_surface_sites;
  sizes->num_ion_exchange_sites = engine->num_ion_exchange_sites;
  sizes->num_gases = engine->num_gases;
  sizes->num_isotherm_species = 0; /* Not directly tracked via PhreeqcRM */
  sizes->num_aqueous_kinetics = RM_GetKineticReactionsCount(engine->rm_id);
  
  if (sizes->num_surface_sites > 0 || sizes->num_ion_exchange_sites > 0) {
    sizes->num_sorbed = sizes->num_primary; /* Follow Alquimia convention */
  } else {
    sizes->num_sorbed = 0;
  }
  
  /* Ensure we allocate exactly 1 integer in auxiliary data to store the 
     negative condition index for the "Time Step 0" cold start workaround. */
  sizes->num_aux_integers = 1;
  sizes->num_aux_doubles = 0;

  /* Setup functionality flags */
  functionality->thread_safe = true;
  functionality->temperature_dependent = true;
  functionality->pressure_dependent = true;
  functionality->porosity_update = false; 
  functionality->operator_splitting = true;
  functionality->global_implicit = false;
  functionality->index_base = 1;

  void** pft_engine_state_ptr = (void**)pft_engine_state;
  *pft_engine_state_ptr = (void*)engine;

  status->error = kAlquimiaNoError;
  snprintf(status->message, kAlquimiaMaxStringLength, "phreeqc_alquimia_setup: Success.");
}

void phreeqc_alquimia_shutdown(void* pft_engine_state,
                               AlquimiaEngineStatus* status) {
  if (pft_engine_state == NULL) {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "phreeqc_alquimia_shutdown: engine_state pointer is NULL.");
    return;
  }
  void** pft_engine_state_ptr = (void**)pft_engine_state;
  if (*pft_engine_state_ptr == NULL) return;

  PhreeqcRMEngineState* engine = (PhreeqcRMEngineState*)(*pft_engine_state_ptr);
  
  int rm_status = RM_Destroy(engine->rm_id);
  if (rm_status < 0) {
    status->error = rm_status;
    snprintf(status->message, kAlquimiaMaxStringLength, "phreeqc_alquimia_shutdown: RM_Destroy failed.");
  } else {
    status->error = kAlquimiaNoError;
    snprintf(status->message, kAlquimiaMaxStringLength, "phreeqc_alquimia_shutdown: Success.");
  }

  free(engine);
  *pft_engine_state_ptr = NULL;
}

void phreeqc_alquimia_processcondition(void* pft_engine_state,
                                       AlquimiaGeochemicalCondition* condition,
                                       AlquimiaProperties* props,
                                       AlquimiaState* state,
                                       AlquimiaAuxiliaryData* aux_data,
                                       AlquimiaEngineStatus* status) {
  if (pft_engine_state == NULL) {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "phreeqc_alquimia_processcondition: engine_state pointer is NULL.");
    return;
  }
  void** pft_engine_state_ptr = (void**)pft_engine_state;
  PhreeqcRMEngineState* engine = (PhreeqcRMEngineState*)(*pft_engine_state_ptr);

  /* Note: PhreeqcRM does not have an API to construct initial conditions dynamically 
     from arrays in memory. Conditions must be predefined in the PHREEQC input file 
     and indexed by number. We check if the driver attempted to pass dynamic 
     constraints and issue an error if they did. */
  if (condition->aqueous_constraints.size > 0 || condition->mineral_constraints.size > 0) {
      status->error = kAlquimiaErrorUnsupportedFunctionality;
      snprintf(status->message, kAlquimiaMaxStringLength, 
               "phreeqc_alquimia_processcondition: PhreeqcRM does not support dynamic memory constraints. "
               "Please define conditions in the PHREEQC input file and use condition names ending in the index (e.g., 'initial_1').");
      return;
  }

  /* A generic condition string typically ends with an index in Alquimia 
     (e.g., 'initial_1'). We attempt to parse that index out, or assume 1. */
  int condition_index = 1;
  const char* p = condition->name + strlen(condition->name);
  while (p > condition->name && isdigit((unsigned char)*(p - 1))) {
      p--;
  }
  if (*p != '\0') {
      condition_index = atoi(p);
  }

  /* Define arrays for InitialPhreeqc2Module (size 1 cell * 7 entities) */
  /* Order: Solution, EqPhase, Exchange, Surface, Gas, Kinetics, SolidSolution */
  int initial_conditions[7] = {-1, -1, -1, -1, -1, -1, -1};
  
  /* We will assign this condition index to all entities for simplicity 
     if they exist, since Phreeqc usually aligns them. */
  initial_conditions[0] = condition_index; // Solution
  if (engine->num_minerals > 0) initial_conditions[1] = condition_index;
  if (engine->num_ion_exchange_sites > 0) initial_conditions[2] = condition_index;
  if (engine->num_surface_sites > 0) initial_conditions[3] = condition_index;
  if (engine->num_gases > 0) initial_conditions[4] = condition_index;

  int rm_status = RM_InitialPhreeqc2Module(engine->rm_id, initial_conditions, NULL, NULL);
  if (rm_status < 0) {
    status->error = rm_status;
    snprintf(status->message, kAlquimiaMaxStringLength, "phreeqc_alquimia_processcondition: RM_InitialPhreeqc2Module failed.");
    return;
  }

  /* Create a unique negative index to represent this specific initial condition 
     in PhreeqcRM's state map, avoiding overlap with positive natural_ids. */
  int state_index = -condition_index;
  RM_StateSave(engine->rm_id, state_index);
  
  /* Store this unique state index in the auxiliary data so the transport code 
     can pass it back to us during ReactionStepOperatorSplit at Time Step 1. */
  if (aux_data->aux_ints.size > 0 && aux_data->aux_ints.data != NULL) {
      aux_data->aux_ints.data[0] = state_index;
  }

  /* Populate the state with the mapped initial conditions */
  rm_status = RM_GetConcentrations(engine->rm_id, state->total_mobile.data);
  if (rm_status == 0) {
    status->error = kAlquimiaNoError;
  } else {
    status->error = rm_status;
  }
}

void phreeqc_alquimia_reactionstepoperatorsplit(void* pft_engine_state,
                                                double delta_t,
                                                AlquimiaProperties* props,
                                                AlquimiaState* state,
                                                AlquimiaAuxiliaryData* aux_data,
                                                int natural_id,
                                                AlquimiaEngineStatus* status) {
  if (pft_engine_state == NULL) {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "phreeqc_alquimia_reactionstepoperatorsplit: engine_state pointer is NULL.");
    return;
  }
  void** pft_engine_state_ptr = (void**)pft_engine_state;
  PhreeqcRMEngineState* engine = (PhreeqcRMEngineState*)(*pft_engine_state_ptr);

  /* 0. Retrieve Initial Guess / State */
  /* Attempt to apply the state from the previous time step using natural_id. */
  int rm_status = RM_StateApply(engine->rm_id, natural_id);
  
  if (rm_status < 0) {
      /* If this fails, the cell has never been saved (Time Step 1). 
         We pull the unique negative state index saved during ProcessCondition. */
      if (aux_data->aux_ints.size > 0 && aux_data->aux_ints.data != NULL) {
          int state_index = aux_data->aux_ints.data[0];
          rm_status = RM_StateApply(engine->rm_id, state_index);
          if (rm_status < 0) {
              /* If this still fails, the transport code somehow didn't call 
                 ProcessCondition properly. We can't recover. */
              status->error = rm_status;
              snprintf(status->message, kAlquimiaMaxStringLength, 
                       "phreeqc_alquimia_reactionstepoperatorsplit: Failed to apply initial state.");
              status->converged = false;
              return;
          }
      }
  }

  /* 1. Setup Time and Step */
  RM_SetTimeStep(engine->rm_id, delta_t);

  /* 2. Setup Properties */
  RM_SetTemperature(engine->rm_id, &state->temperature);
  RM_SetPorosity(engine->rm_id, &state->porosity);
  RM_SetSaturationUser(engine->rm_id, &props->saturation);
  
  /* Density: Alquimia expects kg/m^3. PhreeqcRM uses g/cm^3. */
  double rm_density = state->water_density / 1000.0;
  RM_SetDensityUser(engine->rm_id, &rm_density);
  /* Since RM_UseSolutionDensityVolume is 0, this explicitly holds density constant */

  /* 3. Setup Concentrations */
  /* Mobile: mol / L (mapped 1:1) */
  RM_SetConcentrations(engine->rm_id, state->total_mobile.data);

  /* Note: PhreeqcRM intrinsically remembers and modifies immobile minerals 
     from the Initial condition in memory, so we do not push them manually here 
     unless strictly necessary via a text string. */

  /* 4. Run Chemistry */
  rm_status = RM_RunCells(engine->rm_id);
  if (rm_status < 0) {
    status->error = rm_status;
    snprintf(status->message, kAlquimiaMaxStringLength, "phreeqc_alquimia_reactionstepoperatorsplit: RM_RunCells failed.");
    status->converged = false;
    return;
  }

  /* 5. Save the updated state under the natural_id for the next time step */
  RM_StateSave(engine->rm_id, natural_id);

  /* 6. Retrieve Concentrations */
  RM_GetConcentrations(engine->rm_id, state->total_mobile.data);

  /* Note: We explicitly DO NOT call RM_GetDensityCalculated() to update state->water_density 
     because we want to leave the density exactly as it was provided by the transport code. */

  status->error = kAlquimiaNoError;
  status->converged = true;
  status->num_rhs_evaluations = 1;
  status->num_jacobian_evaluations = 1;
  status->num_newton_iterations = 1;
}

void phreeqc_alquimia_getauxiliaryoutput(void* pft_engine_state,
                                         AlquimiaProperties* props,
                                         AlquimiaState* state,
                                         AlquimiaAuxiliaryData* aux_data,
                                         AlquimiaAuxiliaryOutputData* aux_out,
                                         int natural_id,                                         
                                         AlquimiaEngineStatus* status) {
  if (pft_engine_state == NULL) {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "phreeqc_alquimia_getauxiliaryoutput: engine_state pointer is NULL.");
    return;
  }
  void** pft_engine_state_ptr = (void**)pft_engine_state;
  PhreeqcRMEngineState* engine = (PhreeqcRMEngineState*)(*pft_engine_state_ptr);

  /* Apply the state for the requested cell */
  int rm_status = RM_StateApply(engine->rm_id, natural_id);
  if (rm_status < 0) {
      status->error = rm_status;
      snprintf(status->message, kAlquimiaMaxStringLength, "phreeqc_alquimia_getauxiliaryoutput: RM_StateApply failed for natural_id %d.", natural_id);
      return;
  }

  /* Retrieve Species Data */
  int total_species = engine->num_primary + engine->num_aqueous_complexes;
  
  if (total_species > 0) {
       double* all_species_conc = (double*)malloc(sizeof(double) * total_species);
       double* all_species_gammas = (double*)malloc(sizeof(double) * total_species);
       
       if (all_species_conc && all_species_gammas) {
           RM_GetSpeciesConcentrations(engine->rm_id, all_species_conc);
           RM_GetSpeciesLog10Gammas(engine->rm_id, all_species_gammas);
           
           /* 1. Map Primary Free Ions */
           if (aux_out->primary_free_ion_concentration.data != NULL) {
               for (int i = 0; i < engine->num_primary && i < aux_out->primary_free_ion_concentration.size; ++i) {
                   aux_out->primary_free_ion_concentration.data[i] = all_species_conc[i];
                   if (aux_out->primary_activity_coeff.data != NULL) {
                       aux_out->primary_activity_coeff.data[i] = all_species_gammas[i];
                   }
               }
           }
           
           /* 2. Map Secondary Free Ions */
           if (aux_out->secondary_free_ion_concentration.data != NULL) {
               for (int i = 0; i < engine->num_aqueous_complexes && i < aux_out->secondary_free_ion_concentration.size; ++i) {
                   /* The secondary species follow immediately after the primary species */
                   int phreeqc_index = i + engine->num_primary; 
                   aux_out->secondary_free_ion_concentration.data[i] = all_species_conc[phreeqc_index];
                   if (aux_out->secondary_activity_coeff.data != NULL) {
                       aux_out->secondary_activity_coeff.data[i] = all_species_gammas[phreeqc_index];
                   }
               }
           }
       }
       free(all_species_conc);
       free(all_species_gammas);
  }

  /* Retrieve SELECTED_OUTPUT data (pH, SI, etc) */
  if (engine->num_sel_out_cols > 0) {
      double* sel_out = (double*)malloc(sizeof(double) * engine->num_sel_out_cols);
      RM_GetSelectedOutput(engine->rm_id, sel_out);

      char heading[100];
      for (int i = 0; i < engine->num_sel_out_cols; ++i) {
          RM_GetSelectedOutputHeading(engine->rm_id, i, heading, 100);
          
          if (AlquimiaCaseInsensitiveStringCompare(heading, "pH")) {
              aux_out->pH = sel_out[i];
          }
      }
      free(sel_out);
  }

  status->error = kAlquimiaNoError;
}

void phreeqc_alquimia_getproblemmetadata(void* pft_engine_state,
                                         AlquimiaProblemMetaData* meta_data,
                                         AlquimiaEngineStatus* status) {
  if (pft_engine_state == NULL) {
    status->error = kAlquimiaErrorEngineIntegrity;
    snprintf(status->message, kAlquimiaMaxStringLength, "phreeqc_alquimia_getproblemmetadata: engine_state pointer is NULL.");
    return;
  }
  void** pft_engine_state_ptr = (void**)pft_engine_state;
  PhreeqcRMEngineState* engine = (PhreeqcRMEngineState*)(*pft_engine_state_ptr);

  char name_buffer[100];
  
  if (meta_data->primary_names.data != NULL) {
      for (int i = 0; i < engine->num_primary && i < meta_data->primary_names.size; ++i) {
          RM_GetComponent(engine->rm_id, i, name_buffer, 100);
          if (meta_data->primary_names.data[i]) free(meta_data->primary_names.data[i]);
          meta_data->primary_names.data[i] = AlquimiaStringDup(name_buffer);
      }
  }

  if (meta_data->mineral_names.data != NULL) {
      for (int i = 0; i < engine->num_minerals && i < meta_data->mineral_names.size; ++i) {
          RM_GetEquilibriumPhasesName(engine->rm_id, i, name_buffer, 100);
          if (meta_data->mineral_names.data[i]) free(meta_data->mineral_names.data[i]);
          meta_data->mineral_names.data[i] = AlquimiaStringDup(name_buffer);
      }
  }

  if (meta_data->surface_site_names.data != NULL) {
      for (int i = 0; i < engine->num_surface_sites && i < meta_data->surface_site_names.size; ++i) {
          RM_GetSurfaceName(engine->rm_id, i, name_buffer, 100);
          if (meta_data->surface_site_names.data[i]) free(meta_data->surface_site_names.data[i]);
          meta_data->surface_site_names.data[i] = AlquimiaStringDup(name_buffer);
      }
  }

  if (meta_data->ion_exchange_names.data != NULL) {
      for (int i = 0; i < engine->num_ion_exchange_sites && i < meta_data->ion_exchange_names.size; ++i) {
          RM_GetExchangeName(engine->rm_id, i, name_buffer, 100);
          if (meta_data->ion_exchange_names.data[i]) free(meta_data->ion_exchange_names.data[i]);
          meta_data->ion_exchange_names.data[i] = AlquimiaStringDup(name_buffer);
      }
  }
  
  if (meta_data->gas_names.data != NULL) {
      for (int i = 0; i < engine->num_gases && i < meta_data->gas_names.size; ++i) {
          RM_GetGasComponentsName(engine->rm_id, i, name_buffer, 100);
          if (meta_data->gas_names.data[i]) free(meta_data->gas_names.data[i]);
          meta_data->gas_names.data[i] = AlquimiaStringDup(name_buffer);
      }
  }

  status->error = kAlquimiaNoError;
}
