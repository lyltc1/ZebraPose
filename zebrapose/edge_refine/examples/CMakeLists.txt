cmake_minimum_required(VERSION 3.10)

add_subdirectory(pybind11)

add_executable(evaluate_rbot_dataset evaluate_rbot_dataset.cpp rbot_evaluator.cpp)
target_link_libraries(evaluate_rbot_dataset PUBLIC srt3d)

add_executable(parameter_study_rbot_dataset parameter_study_rbot_dataset.cpp rbot_evaluator.cpp)
target_link_libraries(parameter_study_rbot_dataset PUBLIC srt3d)

add_executable(evaluate_opt_dataset evaluate_opt_dataset.cpp opt_evaluator.cpp)
target_link_libraries(evaluate_opt_dataset PUBLIC srt3d)

add_executable(parameter_study_opt_dataset parameter_study_opt_dataset.cpp opt_evaluator.cpp)
target_link_libraries(parameter_study_opt_dataset PUBLIC srt3d)

add_executable(run_on_recorded_sequence run_on_recorded_sequence.cpp)
    target_link_libraries(run_on_recorded_sequence PUBLIC srt3d)

if(USE_AZURE_KINECT)
    add_executable(record_camera_sequence record_camera_sequence.cpp)
    target_link_libraries(record_camera_sequence PUBLIC srt3d)

    add_executable(run_on_camera_sequence run_on_camera_sequence.cpp)
    target_link_libraries(run_on_camera_sequence PUBLIC srt3d)
endif()

add_executable(understanding_model understanding_model.cpp rbot_evaluator.cpp)  # This is an example for author to understand srt3d project.
target_link_libraries(understanding_model PUBLIC srt3d)

pybind11_add_module(edge_refine edge_refine.cpp rbot_evaluator.cpp)  # This is the main function used by epnet
target_link_libraries(edge_refine PUBLIC srt3d)

pybind11_add_module(edge_refine_debug edge_refine_debug.cpp rbot_evaluator.cpp)  # This is the main function used by epnet
target_link_libraries(edge_refine_debug PUBLIC srt3d)

pybind11_add_module(example example.cpp) # This is an example for author to learn how to use pybind11
