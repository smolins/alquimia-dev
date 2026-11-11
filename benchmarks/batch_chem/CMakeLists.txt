include(add_alquimia_batch_chem_benchmark)

# NOTE: for the moment, the batch chemistry driver doesn't support geochemical 
# NOTE: conditions specified in .cfg files, so we have disabled tests of this
# NOTE: capability (*_ac_*.cfg).

if (ALQUIMIA_HAVE_PFLOTRAN)
#  add_alquimia_batch_chem_benchmark(calcite_short_ac_pflotran pflotran calcite-short-ac-pflotran.cfg)
  add_alquimia_batch_chem_benchmark(calcite_short_pc_pflotran pflotran calcite-short-pc-pflotran.cfg)
  add_alquimia_batch_chem_benchmark(calcite_vf_pc_pflotran pflotran calcite-vf-pc-pflotran.cfg)
#  add_alquimia_batch_chem_benchmark(general_reaction_ac_pflotran pflotran general-reaction-ac-pflotran.cfg)
  add_alquimia_batch_chem_benchmark(general_reaction_pc_pflotran pflotran general-reaction-pc-pflotran.cfg)
#  add_alquimia_batch_chem_benchmark(ion_exchange_valocchi_ac_pflotran pflotran ion-exchange-valocchi-ac-pflotran.cfg)
  add_alquimia_batch_chem_benchmark(ion_exchange_valocchi_pc_pflotran pflotran ion-exchange-valocchi-pc-pflotran.cfg)
#  add_alquimia_batch_chem_benchmark(isotherms_ac_pflotran pflotran isotherms-ac-pflotran.cfg)
  add_alquimia_batch_chem_benchmark(isotherms_pc_pflotran pflotran isotherms-pc-pflotran.cfg)
#  add_alquimia_batch_chem_benchmark(surface_complexation_2_ac_pflotran pflotran surface-complexation-2-ac-pflotran.cfg)
  add_alquimia_batch_chem_benchmark(surface_complexation_2_pc_pflotran pflotran surface-complexation-2-pc-pflotran.cfg)
  add_alquimia_batch_chem_benchmark(microbial_pflotran pflotran microbial-pflotran.cfg)
endif()

if (ALQUIMIA_HAVE_CRUNCHFLOW)
#  add_alquimia_batch_chem_benchmark(calcite_short_ac_crunch crunchflow calcite-short-ac-crunch.cfg)
  add_alquimia_batch_chem_benchmark(calcite_short_cc_crunch crunchflow calcite-short-cc-crunch.cfg)
#  add_alquimia_batch_chem_benchmark(calcite_vf_ac_crunch crunchflow calcite-vf-ac-crunch.cfg)
  add_alquimia_batch_chem_benchmark(calcite_vf_cc_crunch crunchflow calcite-vf-cc-crunch.cfg)
#  add_alquimia_batch_chem_benchmark(general_reaction_ac_crunch crunchflow general-reaction-ac-crunch.cfg)
  add_alquimia_batch_chem_benchmark(general_reaction_cc_crunch crunchflow general-reaction-cc-crunch.cfg)
#  add_alquimia_batch_chem_benchmark(ion_exchange_valocchi_ac_crunch crunchflow ion-exchange-valocchi-ac-crunch.cfg)
  add_alquimia_batch_chem_benchmark(ion_exchange_valocchi_cc_crunch crunchflow ion-exchange-valocchi-cc-crunch.cfg)
#  add_alquimia_batch_chem_benchmark(isotherms_ac_crunch crunchflow isotherms-ac-crunch.cfg)
  add_alquimia_batch_chem_benchmark(isotherms_cc_crunch crunchflow isotherms-cc-crunch.cfg)
#  add_alquimia_batch_chem_benchmark(surface_complexation_2_ac_crunch crunchflow surface-complexation-2-ac-crunch.cfg)
  add_alquimia_batch_chem_benchmark(surface_complexation_2_cc_crunch crunchflow surface-complexation-2-cc-crunch.cfg)
#
  add_alquimia_batch_chem_benchmark(calcite_co2_crunch_cc crunchflow calcite-co2-crunch-cc.cfg)
#
  add_alquimia_batch_chem_benchmark(ch4_o2_crunch_cc crunchflow ch4-o2-crunch-cc.cfg)
endif()

if (ALQUIMIA_HAVE_ONNX)
  add_alquimia_batch_chem_benchmark(ex8_nn_1d onnx ex8-nn-1d.cfg)
  add_alquimia_batch_chem_benchmark(ex8_nn_batch1 onnx ex8-nn-batch1.cfg)
  add_alquimia_batch_chem_benchmark(ex8_nn_dynamic_batch onnx ex8-nn-dynamic-batch.cfg)
  add_alquimia_batch_chem_benchmark(ex8_rf_6_dynamic_batch onnx ex8-rf-6-dynamic-batch.cfg)
  add_alquimia_batch_chem_benchmark(ex8_rf_9_batch1 onnx ex8-rf-9-batch1.cfg)
  add_alquimia_batch_chem_benchmark(ex8_rf_9_dynamic_batch onnx ex8-rf-9-dynamic-batch.cfg)
  add_alquimia_batch_chem_benchmark(ex8_rf_9_1d onnx ex8-rf-9-1d.cfg)
  add_alquimia_batch_chem_benchmark(ex8_rf_9_scalar onnx ex8-rf-9-scalar.cfg)
  foreach(features 1 2 5 6)
    add_alquimia_batch_chem_benchmark(lsurf_${features}_dynamic_batch onnx
      lsurf-${features}-dynamic-batch.cfg
      lsurf_model_${features}_float_64.json
      lsurf_model_${features}_float_64.onnx)
  endforeach()

endif()
