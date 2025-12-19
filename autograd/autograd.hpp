#include <iostream> 
#include <vector> 
#include <array>
#include <iomanip> 
#include <functional>
#include <assert.h> 
#include <cmath> 

std::mt19937 engine(42);
double uniform(double l, double r) {
    std::uniform_real_distribution<double> dist(l, r); 
    return dist(engine); 
}

struct Node {
    double x, grad;
    bool is_parameter; 
    std::array<int, 2> child_id;
    std::function<void(Node&, Node&, Node&)> backward_func; 

    Node(
        double _x = 0.0, 
        double _grad = 0.0, 
        std::array<int, 2> _child_id = {-1, -1}, 
        std::function<void(Node&, Node&, Node&)> _backward_func = NULL, 
        double _is_parameter = 0
    ) : x(_x), grad(_grad), child_id(_child_id), backward_func(_backward_func), is_parameter(_is_parameter) {} 

    void set_parameter() {
        is_parameter = 1; 
    }
}; 

struct Tape {
    std::vector<Node> nodes; 
    int add_node(Node n) {
        nodes.emplace_back(std::move(n)); 
        return (int)nodes.size() - 1; 
    }

    void backward(int s) {
        std::vector<int> topo; 
        std::vector<int> vis((int)nodes.size(), 0); 
        auto&& dfs = [&](this auto&& dfs, int u) -> void {
            vis[u] = 1; 
            for (int i = 0; i < 2; i++) {
                int v = nodes[u].child_id[i]; 
                if (v > -1 && !vis[v]) {
                    dfs(v); 
                } 
            }
            topo.emplace_back(u); 
        }; 
        dfs(s); 
        reverse(topo.begin(), topo.end()); 

        nodes[s].grad = 1; 
        for (int u : topo) {
            if (nodes[u].backward_func == NULL) continue; 

            int l = nodes[u].child_id[0]; 
            int r = nodes[u].child_id[1]; 

            // possibly not the best to pass nodes[l] 2 times for unary functions
            // but I don't see it breaking at least for now
            nodes[u].backward_func(nodes[u], nodes[l], (r > -1 ? nodes[r] : nodes[l]));
        }
    }
}; 
thread_local Tape tape; 

struct Value {
    int id; 
    Value(double x, bool is_parameter = 0) {
        Node node(x); 
        if (is_parameter) node.set_parameter(); 
        id = tape.add_node(std::move(node));
    }
    Value(Node&& node, bool is_parameter = 0) {
        if (is_parameter) node.set_parameter(); 
        id = tape.add_node(std::move(node)); 
    }
    Value() {} 

    Value operator+(const Value& ot) const {
        Node &a = tape.nodes[id], &b = tape.nodes[ot.id]; 
        Node par_node(a.x + b.x, 0.0, {id, ot.id}, [](Node& par, Node& a, Node& b) -> void {
            a.grad += par.grad; 
            b.grad += par.grad; 
        }); 
        return Value(std::move(par_node)); 
    }
    Value operator*(const Value& ot) const {
        Node &a = tape.nodes[id], &b = tape.nodes[ot.id]; 
        Node par_node(a.x * b.x, 0.0, {id, ot.id}, [](Node &par, Node &a, Node& b) -> void {
            a.grad += par.grad * b.x; 
            b.grad += par.grad * a.x; 
        }); 
        return Value(std::move(par_node)); 
    }
    Value operator-() const {
        Node &a = tape.nodes[id]; 
        Node par_node(-a.x, 0.0, {id, -1}, [](Node &par, Node &a, Node &b) -> void {
            a.grad -= par.grad; 
        }); 
        return Value(std::move(par_node)); 
    }
    Value operator-(const Value& ot) const {
        return (*this) + (-ot); 
    }

    // have to change functions back to lambda, generalizing them too much doens't work 
    // here, for example, it needs to be aware of the exponent c
    Value pow(const double c) const {
    //        this -> grad += parent_grad * (c * std::pow(this -> x, c - 1)); 
        Node &a = tape.nodes[id]; 
        Node par_node(std::pow(a.x, c), 0.0, {id, -1}, [c](Node &par, Node &a, Node &b) -> void {
            a.grad += par.grad * (c * std::pow(a.x, c - 1)); 
        }); 
        return Value(std::move(par_node)); 
    }

    Value operator/(const Value& ot) const {
        return (*this) * (ot.pow(-1)); 
    }

    Value exp() const {
        Node &a = tape.nodes[id]; 
        Node par_node(std::exp(a.x), 0.0, {id, -1}, [](Node &par, Node &a, Node &b) -> void {
            a.grad += par.grad * std::exp(a.x); 
        }); 
        return Value(std::move(par_node)); 
    }

    Value tanh() const {
        Value e_to_x = this -> exp(); 
        Value e_to_m_x = (-(*this)).exp(); 

        Node e = tape.nodes[e_to_x.id]; 
        Node me = tape.nodes[e_to_m_x.id]; 

        return (e_to_x - e_to_m_x) / (e_to_x + e_to_m_x); 
    }

    Value log() const {
        Node &a = tape.nodes[id]; 
        Node par_node(std::log(a.x), 0.0, {id, -1}, [](Node &par, Node &a, Node &b) -> void {
            a.grad += par.grad / a.x; 
        }); 
        return Value(std::move(par_node)); 
    }

    void backward() {
        tape.backward(id); 
    }
};  

std::ostream& operator<<(std::ostream& os, const Value& v) {
    os << "Value(x=" << tape.nodes[v.id].x << ", grad=" << tape.nodes[v.id].grad << ")"; 
    return os; 
}

struct Neuron {
    std::vector<Value> w; 
    Value b; 
    bool no_bias, no_activation; 
    Neuron(int n_in, bool _no_bias = 0, bool _no_activation = 0) : no_bias(_no_bias), no_activation(_no_activation) {
        w.reserve(n_in); 
        for (int i = 0; i < n_in; i++) w.emplace_back(Value(uniform(-1, 1), 1)); 
        if (!no_bias) 
            b = Value(uniform(-1, 1), 1); 
    } 
    Value operator()(const std::vector<Value> x) const {
        assert(w.size() == x.size()); 
        Value ret = no_bias ? 0 : b; 
        for (int i = 0; i < (int)w.size(); i++) ret = ret + w[i]*x[i]; 
        return no_activation ? ret : ret.tanh(); 
    }

    std::vector<Value> parameters() {
        std::vector<Value> params = w; 
        if (!no_bias)
            params.emplace_back(b); 
        return params; 
    }
}; 

struct Layer {
    std::vector<Neuron> neurons; 
    Layer() {}
    Layer(int n_in, int n_out, bool no_bias = 0, bool no_activation = 0) {
        assert(n_in > 0 && n_out > 0); 
        neurons.reserve(n_out); 
        for (int i = 0; i < n_out; i++) neurons.emplace_back(Neuron(n_in, no_bias, no_activation)); 
    }
    std::vector<Value> operator()(const std::vector<Value>& x) const {
        assert(x.size() == neurons[0].w.size()); 
        std::vector<Value> ret((int)neurons.size(), 0); 
        for (int i = 0; i < (int)neurons.size(); i++) {
            ret[i] = neurons[i](x); 
        }
        return ret; 
    }

    std::vector<Value> parameters() {
        std::vector<Value> params; 
        for (Neuron neuron : neurons) {
            std::vector<Value> neuron_params = neuron.parameters(); 
            params.insert(params.end(), neuron_params.begin(), neuron_params.end()); 
        }
        return params; 
    }
}; 

std::vector<Value> oneHot(int n, int pos) {
    std::vector<Value> ret(n, 0); 
    ret[pos] = 1; 
    return ret; 
}

struct Linear {
    Layer layer; 
    Linear(int n_in, int n_out, bool no_bias = 0) {
        layer = Layer(n_in, n_out, no_bias, 1); 
    }
    std::vector<Value> operator()(const std::vector<Value>& x) const {
        return layer(x); 
    }
    std::vector<Value> parameters() {
        return layer.parameters(); 
    }
}; 

struct MLP {
    std::vector<Layer> layers; 
    MLP(int n_in, const std::vector<int> layer_sizes) {
        layers.reserve(1 + (int)layer_sizes.size()); 
        layers.emplace_back(n_in, layer_sizes[0]); 
        for (int i = 0; i + 1 < (int)layer_sizes.size(); i++) {
            layers.emplace_back(layer_sizes[i], layer_sizes[i+1]); 
        }
    }
    std::vector<Value> operator()(const std::vector<Value>& x) const {
        std::vector<Value> cur_x = x; 
        for (int i = 0; i < (int)layers.size(); i++) {
            cur_x = layers[i](cur_x); 
        }
        return cur_x; 
    }

    std::vector<Value> parameters() {
        std::vector<Value> params; 
        for (Layer layer : layers) {
            std::vector<Value> layer_params = layer.parameters(); 
            params.insert(params.end(), layer_params.begin(), layer_params.end()); 
        }
        return params; 
    }
}; 