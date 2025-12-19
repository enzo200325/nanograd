#include <iostream> 
#include <iomanip> 
#include <cmath> 
#include <random> 
#include <chrono> 
#include <vector>
#include "autograd.hpp"
#include <nlohmann/json.hpp>
#include <fstream> 
#include <typeinfo> 
#include <string> 
using json = nlohmann::json; 

using namespace std; 

void print(const std::vector<Value> &vec) {
    std::cout << "{" << std::endl; 
    for (const Value val : vec) {
        std::cout << "\t" << val << ", " << std::endl;
    }
    std::cout << "}" << std::endl; 
}

int uniform(int l, int r) {
    uniform_int_distribution dist(l, r); 
    return dist(engine); 
}

struct EmbeddingTable {
    int n, m; 
    vector<vector<Value>> table; 
    EmbeddingTable(int _n, int _m) : n(_n), m(_m) {
        table.assign(n, vector<Value>(m)); 
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                table[i][j] = Value(uniform(-1.0, 1.0)); 
            }
        }
    }
    vector<Value> get_idx(int i) {
        return table[i]; 
    }
    vector<vector<Value>> get_idxs(vector<int> I) {
        vector<vector<Value>> ret; 
        for (int i : I) ret.emplace_back(table[i]); 
        return ret; 
    }
}; 

typedef vector<vector<vector<Value>>> v3; 
typedef vector<vector<Value>> v2; 
typedef vector<Value> v1; 

void init_zero(v2& a, bool is_parameter = 0) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[i].size(); j++) {
            a[i][j] = Value(0.0, is_parameter); 
        }
    }
}
void init_random(v2& a, bool is_parameter = 0, double l = -.1, double r = .1) {
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[i].size(); j++) {
            a[i][j] = Value(uniform(l, r), is_parameter); 
        }
    }
}

v2 transpose(const v2& a) {
    v2 ret; 
    ret.assign(a[0].size(), v1(a.size())); 
    for (int i = 0; i < a[0].size(); i++) {
        for (int j = 0; j < a.size(); j++) {
            ret[i][j] = a[j][i]; 
        }
    }
    return ret; 
}

v2 operator*(const v2& a, const v2& b) {
    assert((int)a[0].size() == b.size()); 
    v2 ret((int)a.size(), v1((int)b[0].size())); 
    init_zero(ret); 
    for (int i = 0; i < (int)a.size(); i++) {
        for (int j = 0; j < (int)b[0].size(); j++) {
            for (int k = 0; k < (int)a[0].size(); k++) {
                ret[i][j] = ret[i][j] + a[i][k] * b[k][j]; 
            }
        }
    }
    return ret; 
}

v2 operator+(const v2& a, const v2& b) {
    v2 ret(a.size(), v1(a[0].size())); 
    init_zero(ret); 
    for (int i = 0; i < (int)a.size(); i++) {
        assert(a[i].size() == b[i].size()); 
        for (int j = 0; j < a[i].size(); j++) {
            ret[i][j] = a[i][j] + b[i][j]; 
        }
    }
    return ret; 
}

v1 operator+(const v1& a, const v1& b) {
    assert(a.size() == b.size()); 
    v1 ret(a.size()); 
    for (int i = 0; i < (int)a.size(); i++) {
        ret[i] = a[i] + b[i]; 
    }
    return ret; 
}

void softmax(v1 &vec) {
    Value deno(0.0); 
    for (int i = 0; i < (int)vec.size(); i++) deno = deno + vec[i].exp(); 
    for (int i = 0; i < (int)vec.size(); i++) vec[i] = vec[i].exp() / deno; 
}
void softmax(v2 &mat) {
    for (int i = 0; i < mat.size(); i++) softmax(mat[i]); 
}

const int B = 2; 
const int T = 16; 
const int C = 32;  

int main() {
    std::ifstream raw_data_file("./python-codes-25k.json"); 
    json data_file = json::parse(raw_data_file); 

    int total_N = data_file.size(); 
    vector<string> raw_examples; 
    for (int i = 0; i < total_N; i++) {
        string cur_example = data_file[i]["output"]; 
        //string cur_example = "abcdefghijklmnopqrstuvwxyz"; 
        raw_examples.emplace_back(cur_example); 
    }


    vector<char> alphabet; 
    for (string cur_example : raw_examples) {
        for (char c : cur_example) alphabet.emplace_back(c); 
    }
    sort(alphabet.begin(), alphabet.end()); 
    alphabet.erase(unique(alphabet.begin(), alphabet.end()), alphabet.end()); 
    int alphabet_size = alphabet.size(); 

    unordered_map<char, int> char_to_idx; 
    for (int i = 0; i < alphabet_size; i++) {
        char_to_idx[alphabet[i]] = i; 
    }

    auto decode = [&](int i) -> char {
        return alphabet[i]; 
    }; 
    auto encode = [&](char c) -> int {
        return char_to_idx[c]; 
    }; 

    const int init_token = encode('\n'); 
    vector<int> possible_xs; 
    vector<int> possible_ys; 
    bool f = 0; 
    for (string example : raw_examples) {
        vector<int> example_as_int; 
        for (char c : example) example_as_int.emplace_back(encode(c)); 
        if (!f) {
            f = 1; 
            cout << "example: " << example << endl;
            cout << "int_example: " << endl;
            for (int v : example_as_int) cout << v << " "; 
            cout << endl;
        }
        for (int v : example_as_int) possible_xs.emplace_back(v); 
    }
    for (int i = 1; i < possible_xs.size(); i++) possible_ys.emplace_back(possible_xs[i]); 
    possible_ys.emplace_back(init_token); 

    Linear embedding((int)alphabet.size(), C, 1); 
    Linear positionalEncoding(T, C, 1); 
    Linear W_q(C, C, 1), W_k(C, C, 1), W_v(C, C, 1); 
    MLP mlp(C, {2*C, 2*C, C}); 
    Linear w_logits(C, (int)alphabet.size(), 1); 

    auto cnt_params = [&]() -> int {
        int cnt_parameters = 0;
        for (Node parameter : tape.nodes) {
            if (parameter.is_parameter)
                cnt_parameters++;
        }
        return cnt_parameters; 
    }; 
    cout << "WTF: " << cnt_params() << endl;

    cout << "alphabet size: " << alphabet.size() << endl;

    auto get_batch = [&]() -> pair<v2, vector<int>>
    {
        int idx = uniform(0, (int)possible_xs.size() - T); 
        v2 ret; 
        vector<int> ys_ret; 
        for (int i = 0; i < T; i++)
        {
            ret.emplace_back(embedding(oneHot((int)alphabet.size(), possible_xs[idx + i]))); 
            ys_ret.emplace_back(possible_ys[idx + i]); 
        }
        return make_pair(ret, ys_ret);
    };



    for (int iterations = 0; iterations < 1000; iterations++)
    {
        while (!tape.nodes.back().is_parameter) tape.nodes.pop_back(); 
        for (Node& param : tape.nodes) {
            param.grad = 0; 
        }

        vector<vector<int>> ys;
        ys.assign(B, vector<int>());
        v3 x; // input (B x T x C)
        for (int i = 0; i < B; i++)
        {
            auto [batch, cur_ys] = get_batch();
            x.emplace_back(batch);
            ys[i] = cur_ys;
        }

        // adds positional encoding
        for (int i = 0; i < B; i++) {
            for (int j = 0; j < T; j++) {
                x[i][j] = x[i][j] + positionalEncoding(oneHot(T, j));
            } 
        }

        v3 Q, K, V;
        Q.assign(B, v2(T, v1(C)));
        K.assign(B, v2(T, v1(C))); // K^t
        V.assign(B, v2(T, v1(C)));
        for (int i = 0; i < B; i++) {
            for (int j = 0; j < T; j++) {
                Q[i][j] = W_q(x[i][j]); 
                K[i][j] = W_k(x[i][j]); 
                V[i][j] = W_v(x[i][j]); 
            } 
        }

        v3 QK;
        QK.assign(B, v2(T, v1(T)));
        for (int i = 0; i < B; i++)
        {
            init_zero(QK[i]);
            QK[i] = Q[i] * transpose(K[i]);

            for (int j = 0; j < T; j++) {
                for (int k = j + 1; k < T; k++) {
                    QK[i][j][k] = Value(-1e9); 
                }
                for (int k = 0; k <= j; k++) QK[i][j][k] = QK[i][j][k] / Value(C).pow(0.5); 
            }

            softmax(QK[i]);
            x[i] = x[i] + QK[i] * V[i];
        }

        for (int i = 0; i < B; i++) {
            for (int j = 0; j < T; j++) {
                x[i][j] = mlp(x[i][j]); 
            }
        }

        for (int i = 0; i < B; i++) {
            for (int j = 0; j < T; j++) {
                x[i][j] = w_logits(x[i][j]); 
            }

            softmax(x[i]);
            Value cur_loss(0.0);
            for (int j = 0; j < T; j++)
            {
                cur_loss = cur_loss + x[i][j][ys[i][j]].log();
            }
            cur_loss = cur_loss / T; 

            cur_loss = -cur_loss;
            cur_loss.backward();
            cout << "i: " << i << " | loss: " << cur_loss << endl;
        }
        cout << endl;

        int it = 0; 
        double lr = 0.01; 
        //if (iterations > 20) lr = 0.5; 
        //if (iterations > 50) lr = 0.2; 
        //if (iterations > 100) lr = 0.1; 
        double clip = 1.0;
        for (Node& parameter : tape.nodes)
        {
            if (parameter.is_parameter) {
                //cout << "prex: " << parameter.x << endl;
                parameter.x -= lr * parameter.grad / B; 
                //cout << "posx: " << parameter.x << endl;
                //cout << endl;
            }
            it++; 
            // cout << "x: " << parameter.x << " | grad: " << parameter.grad << endl;
        }
    }

    vector<double> vals; 
    for (Node param : tape.nodes) if (param.is_parameter) {
        vals.emplace_back(param.x); 
    }
    sort(vals.rbegin(), vals.rend()); 
    cout << "greatest vals: " << endl;
    for (int i = 0; i < 20; i++) cout << vals[i] << " "; 
    cout << endl;






    // --- GERANDO TEXTO (INFERÊNCIA) ---
    cout << "\n--- GERANDO TEXTO (GREEDY) ---" << endl;

    // Começamos com um prefixo (os primeiros caracteres do alfabeto ou um token inicial)
    vector<int> generated_indices;
    generated_indices.push_back(encode('a')); // Inicia com 'a'

    for (int length = 0; length < 40; length++)
    {
        // 1. LIMPEZA DA TAPE: Removemos nós que não são parâmetros para economizar memória
        while (!tape.nodes.empty() && !tape.nodes.back().is_parameter)
        {
            tape.nodes.pop_back();
        }

        // 2. PREPARAÇÃO DO CONTEXTO (Janela de tamanho T)
        v2 x_gen;
        int current_len = generated_indices.size();

        // Pegamos os últimos T caracteres. Se houver menos que T, fazemos "padding" com init_token
        for (int j = 0; j < T; j++)
        {
            int target_idx = current_len - T + j;
            if (target_idx >= 0)
            {
                x_gen.emplace_back(embedding(oneHot(alphabet_size, generated_indices[target_idx])));
            }
            else
            {
                x_gen.emplace_back(embedding(oneHot(alphabet_size, init_token)));
            }
        }

        // 3. POSITIONAL ENCODING
        for (int j = 0; j < T; j++)
        {
            x_gen[j] = x_gen[j] + positionalEncoding(oneHot(T, j));
        }

        // 4. ATTENTION HEAD
        v2 Q(T, v1(C)), K(T, v1(C)), V(T, v1(C));
        for (int j = 0; j < T; j++)
        {
            Q[j] = W_q(x_gen[j]);
            K[j] = W_k(x_gen[j]);
            V[j] = W_v(x_gen[j]);
        }

        v2 QK = Q * transpose(K);
        // Mascaramento Causal e Escalonamento
        for (int j = 0; j < T; j++)
        {
            for (int k = j + 1; k < T; k++)
            {
                QK[j][k] = Value(-1e9);
            }
            for (int k = 0; k <= j; k++)
            {
                QK[j][k] = QK[j][k] / Value(sqrt(C));
            }
        }
        softmax(QK);

        // Conexão Residual: x = x + Attention(x)
        v2 att_out = QK * V;
        for (int j = 0; j < T; j++)
        {
            x_gen[j] = x_gen[j] + att_out[j];
        }

        // 5. MLP (Feed Forward)
        for (int j = 0; j < T; j++)
        {
            x_gen[j] = mlp(x_gen[j]);
        }

        // 6. LOGITS E SELEÇÃO (Pegamos apenas a última posição da janela T)
        v1 last_logits = w_logits(x_gen[T - 1]);
        softmax(last_logits);

        // Seleção Greedy (Maior Probabilidade)
        int best_idx = 0;
        double max_p = -1e18;
        for (int k = 0; k < alphabet_size; k++)
        {
            if (tape.nodes[last_logits[k].id].x > max_p)
            {
                max_p = tape.nodes[last_logits[k].id].x;
                best_idx = k;
            }
        }

        generated_indices.push_back(best_idx);
        cout << decode(best_idx);
        cout.flush(); // Imprime caractere por caractere

        if (decode(best_idx) == init_token)
            break; // Para se gerar o newline
    }
    cout << "\n--- FIM ---" << endl;

    return 0; 
} 
