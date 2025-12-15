cmake_minimum_required(VERSION 3.23)
set (CMAKE_CXX_STANDARD 23)

project(nanograd)

add_executable(myProgram)
target_sources(myProgram
    PRIVATE
        main.cpp
)

add_library(autograd INTERFACE)

target_include_directories(autograd
    INTERFACE
        autograd # Adiciona o subdiretório 'autograd'
)


target_sources(autograd
    INTERFACE
        autograd
)

target_compile_features(autograd
    INTERFACE
        cxx_std_23
)

target_link_libraries(myProgram
    PRIVATE 
        autograd 
)

