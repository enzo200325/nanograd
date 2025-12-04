cmake_minimum_required(VERSION 3.23)
set (CMAKE_CXX_STANDARD 23)

project(nzoCad)

add_executable(myProgram)
target_sources(myProgram
    PRIVATE
        main.cpp
)

