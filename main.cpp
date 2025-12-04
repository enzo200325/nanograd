#include <iostream> 
#include <vector> 
#include <array>
#include <iomanip> 
#include <functional>
#include <vector>

std::function<void(double)> default_backward_func = [](double x) {};

struct Value {
    double x, grad; 
    std::array<Value*, 2> child; 
    std::function<void(double)> backward_func; 

    Value(
        double _x = 0.0, 
        Value* child_1 = nullptr, 
        Value* child_2 = nullptr, 
        std::function<void(double)> _backward_func = default_backward_func, 
        double _grad = 0
    ) : x(_x), child({child_1, child_2}), backward_func(_backward_func), grad(_grad) {} 

    Value* operator+(Value& ot) {
        std::function<void(double)> parent_backward_function = [&](double parent_grad) -> void {
            this -> grad += parent_grad; 
            ot.grad += parent_grad; 
        }; 
        Value* parent = new Value(this -> x + ot.x, this, &ot, parent_backward_function); 
        return parent; 
    }

    Value* operator-() {
        std::function<void(double)> parent_backward_function = [&](double parent_grad) -> void {
            this -> grad += -parent_grad; 
        }; 
        Value* parent = new Value(-x, this, nullptr, parent_backward_function);  
        return parent; 
    }

    Value* operator-(Value& ot) {
        return (*this) + *(-(ot)); 
    }

    Value* operator*(Value& ot) {
        std::function<void(double)> parent_backward_function = [&](double parent_grad) -> void {
            this -> grad += parent_grad * ot.x; 
            ot.grad += parent_grad * (this -> x); 
        }; 
        Value* parent = new Value(this -> x * ot.x, this, &ot, parent_backward_function); 
        return parent; 
    }

    void backward() {
        std::vector<Value*> topo; 
        auto&& dfs = [&](this auto&& dfs, Value* u) -> void {
            int cnt = 0; 
            for (int i = 0; i < 2; i++) if (u -> child[i] != nullptr) dfs(u -> child[i]), cnt++; 
            topo.emplace_back(u); 
        }; 
        dfs(this); 
        reverse(topo.begin(), topo.end()); 

        topo[0] -> grad = 1; 
        for (Value* val : topo) {
            val -> backward_func(val -> grad); 
        } 
    }

    friend std::ostream& operator<<(std::ostream& os, const Value& v) {
        os << "Value(x=" << v.x << ", grad=" << v.grad << ")"; 
        return os; 
    }
}; 

int main() {
    Value a = Value(2); 
    Value b = Value(3); 
    Value* c = a * b; 
    c -> backward(); 

    std::cout << a << std::endl; 
    std::cout << b << std::endl;
    std::cout << *c << std::endl;
}