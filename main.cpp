#include <iostream> 
#include <iomanip> 
#include <cmath> 
#include <random> 
#include <chrono> 
#include <vector>
#include "autograd.hpp"

void print(const std::vector<Value> &vec) {
    std::cout << "{" << std::endl; 
    for (const Value val : vec) {
        std::cout << "\t" << val << ", " << std::endl;
    }
    std::cout << "}" << std::endl; 
}

int main() {


    std::vector<std::vector<Value>> xs; 
    xs.emplace_back(std::vector<Value>{2.0, 3.0, -1.0}); 
    xs.emplace_back(std::vector<Value>{3.0, -1.0, 0.5}); 
    xs.emplace_back(std::vector<Value>{0.5, 1.0, 1.0}); 
    xs.emplace_back(std::vector<Value>{1.0, 1.0, -1.0}); 
    std::vector<Value> ys = {1.0, -1.0, -1.0, 1.0}; 

    std::cout << (int)tape.nodes.size() << std::endl;

    MLP mlp = MLP(3, {4, 4, 1}); 

    int cnt = 10000; 
    while (cnt--) {
        for (Value param : mlp.parameters()) {
            tape.nodes[param.id].grad = 0.0; 
        }

        std::vector<Value> res; 
        res.reserve((int)xs.size()); 
        for (auto x : xs) {
            res.emplace_back(mlp(x)[0]); 
        }

        print(res); 

        Value loss = 0; 
        for (int i = 0; i < res.size(); i++) {
            loss = loss + (res[i] - ys[i]) * (res[i] - ys[i]); 
        }

        loss.backward(); 

        std::cout << "cnt: " << cnt << " | " << "loss: " << loss << std::endl;

        for (Value param : mlp.parameters()) {
            tape.nodes[param.id].x -= tape.nodes[param.id].grad * 0.1; 
        }
    } 
} 