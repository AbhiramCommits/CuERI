set(CUEST_TEST_LINK cuest_eri)

function(cuest_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE ${CUEST_TEST_LINK})
  target_include_directories(${name} PRIVATE
      ${CMAKE_SOURCE_DIR}/include
      ${CMAKE_SOURCE_DIR}/src)
  target_compile_definitions(${name} PRIVATE
      CUEST_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
      CUEST_EXPECT_CPU_ONLY=$<BOOL:${CUEST_CUDA_AVAILABLE}>0)
  target_compile_options(${name} PRIVATE -Wall -Wextra)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES TIMEOUT 1800)
endfunction()

cuest_add_test(test_api)
cuest_add_test(test_boys)
cuest_add_test(test_one_electron)
cuest_add_test(test_eri)
