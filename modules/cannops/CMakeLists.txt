  # libacl_op_compiler.so
  set(lib_acl_op_compiler "${CANN_INSTALL_DIR}/lib64")
  find_library(found_lib_acl_op_compiler NAMES acl_op_compiler PATHS ${lib_acl_op_compiler} NO_DEFAULT_PATH)
  if(found_lib_acl_op_compiler)
      set(lib_acl_op_compiler ${found_lib_acl_op_compiler})
      message(STATUS "CANN: libacl_op_compiler.so is found at ${lib_acl_op_compiler}")
  else()
      message(STATUS "CANN: Missing libacl_op_compiler.so. Turning off HAVE_CANN")
      set(HAVE_CANN OFF)
      return()
  endif()
 if(IOS OR WINRT OR ANDROID OR APPLE OR WIN32 OR (NOT HAVE_CANN))
   ocv_module_disable(cannops)
 endif()

set(the_description "Ascend-accelerated Operations on Matrices")

ocv_add_module(cannops opencv_core WRAP python)
ocv_module_include_directories(${CANN_INCLUDE_DIRS})
ocv_glob_module_sources()
ocv_install_used_external_targets(${CANN_LIBRARIES} ${lib_acl_op_compiler})
ocv_create_module(${CANN_LIBRARIES} ${lib_acl_op_compiler})

ocv_include_directories(${CMAKE_SOURCE_DIR}/modules/ts/include)

ocv_add_accuracy_tests(DEPENDS_ON opencv_cannops)
ocv_add_perf_tests(DEPENDS_ON opencv_cannops)
ocv_add_samples(opencv_cannops)
