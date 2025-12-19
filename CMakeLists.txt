cmake_minimum_required(VERSION 3.23)
set(CMAKE_CXX_STANDARD 23)

project(nanograd)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# Criar uma opção para ligar/desligar facilmente
option(ENABLE_SANITIZER "Habilitar Address Sanitizer" ON)

if(ENABLE_SANITIZER)
    if(MSVC)
        # Para Visual Studio (Windows)
        add_compile_options(/fsanitize=address)
        add_link_options(/fsanitize=address)
    else()
        # Para GCC e Clang (Linux/Mac)
        add_compile_options(-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address -fsanitize=undefined)
    endif()
endif()

# 1. Carregar o módulo FetchContent
include(FetchContent)

# 2. Declarar a biblioteca nlohmann_json
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(json)

# --- Seu alvo autograd ---
add_library(autograd INTERFACE)

target_include_directories(autograd
    INTERFACE
        autograd
)

target_compile_features(autograd
    INTERFACE
        cxx_std_23
)

# --- Seu executável ---
add_executable(myProgram main.cpp)

# 3. Vincular as bibliotecas
target_link_libraries(myProgram
    PRIVATE 
        autograd
        nlohmann_json::nlohmann_json  # <--- Adicionado aqui
)
