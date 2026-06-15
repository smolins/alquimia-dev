/* -*-  mode: c++; c-default-style: "google"; indent-tabs-mode: nil -*- */

#ifndef PHREEQC_ALQUIMIA_INTERFACE_H_
#define PHREEQC_ALQUIMIA_INTERFACE_H_

#include "alquimia/alquimia_containers.h"

#ifdef __cplusplus
extern "C" {
#endif

void phreeqc_alquimia_setup(const char* input_filename,
                            bool hands_off,
                            void* engine_state,
                            AlquimiaSizes* sizes,
                            AlquimiaEngineFunctionality* functionality,
                            AlquimiaEngineStatus* status);

void phreeqc_alquimia_shutdown(void* engine_state,
                               AlquimiaEngineStatus* status);

void phreeqc_alquimia_processcondition(void* engine_state,
                                       AlquimiaGeochemicalCondition* condition,
                                       AlquimiaProperties* props,
                                       AlquimiaState* state,
                                       AlquimiaAuxiliaryData* aux_data,
                                       AlquimiaEngineStatus* status);

void phreeqc_alquimia_reactionstepoperatorsplit(void* engine_state,
                                                double delta_t,
                                                AlquimiaProperties* props,
                                                AlquimiaState* state,
                                                AlquimiaAuxiliaryData* aux_data,
                                                int natural_id,
                                                AlquimiaEngineStatus* status);

void phreeqc_alquimia_getauxiliaryoutput(void* engine_state,
                                         AlquimiaProperties* props,
                                         AlquimiaState* state,
                                         AlquimiaAuxiliaryData* aux_data,
                                         AlquimiaAuxiliaryOutputData* aux_out,
                                         AlquimiaEngineStatus* status);

void phreeqc_alquimia_getproblemmetadata(void* engine_state,
                                         AlquimiaProblemMetaData* meta_data,
                                         AlquimiaEngineStatus* status);

#ifdef __cplusplus
}
#endif

#endif  /* PHREEQC_ALQUIMIA_INTERFACE_H_ */
