#include "autograd.h"
#include <iostream> 
#include <random> 

std::mt19937 engine(42);
double uniform(double l, double r) {
    std::uniform_real_distribution<double> dist(l, r); 
    return dist(engine); 
}

struct Neuron {
    vector<autograd::Value*> w; 
    autograd::Value* b; 

    Neuron(vector<autograd::Value*> _w, autograd::Value* _b) : w(_w), b(_b) {} 
    autograd::Value* operator() {

    }
}

int main() {

}
