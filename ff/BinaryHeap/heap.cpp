#include <iostream>
#include <vector>

typedef bool(*predicate)(const int&, const int&);
typedef const int& const_reference;

bool min_heap(const int& a, const int& b) { return a < b; }
bool max_heap(const int& a, const int& b) { return a > b; }

// можно ещё шаблон относительно типа данных ввести, но не хочется
template <predicate pr = &min_heap>
class heap {
    std::vector<int> p;
public:
    heap() {  }
    heap(std::initializer_list<int> init_list) {
        for (const int& el : init_list) {
            add(el);
        }
    }
    ~heap() {
        clear();
    }
    void clear() {
        p.clear();
    }
    bool empty() {
        return p.empty();
    }
    void pop() {
        std::swap(p[0], p[p.size() - 1]);
        p.pop_back();
        heapify(0);
    }
    const_reference top() {
        return p[0];
    }

    void add(const int& value) {
        p.push_back(value);
        int n = p.size();
        
        for (int i = n - 1; i > 0; i = (i - 1) / 2) {
            if (!pr(p[i], p[(i - 1) / 2])) {
                break;
            }
            std::swap(p[i], p[(i - 1) / 2]);
        }
    }

    void print_heap() {
        for (const int& el : p) {
            std::cout << el << " ";
        }
        std::cout << "\n";
    }
private:
    void heapify(const int& ind) {

        int l = 2 * ind + 1,
            r = 2 * ind + 2;

        int ind_of_el_to_swap_with = ind;
        
        if (pr(p[r], p[ind_of_el_to_swap_with]) && r < p.size()) {
            ind_of_el_to_swap_with = r;
        }
        if (pr(p[l], p[ind_of_el_to_swap_with]) && l < p.size()) {
            ind_of_el_to_swap_with = l;
        }

        if (ind != ind_of_el_to_swap_with) {
            std::swap(p[ind], p[ind_of_el_to_swap_with]);
            heapify(ind_of_el_to_swap_with);
        }
    }
};

int main() {

    heap<max_heap> h = {12, 43, 14, 28};

    h.add(12);
    h.add(19);
    h.add(14);
    h.add(7);
    h.add(15);

    std::cout << h.top() << "\n";
    h.print_heap();

    h.pop();
    std::cout << h.top() << "\n";
    h.print_heap();

    return 0;
}