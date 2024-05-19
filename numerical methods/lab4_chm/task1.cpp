#include <iostream>
#include <vector>
#include <cmath>

#define pi 3.141592653589793

using namespace std;

bool condition(const vector<double>& _v1, const vector<double>& _v2, const double& _eps) {
    if (_v1.size() != _v2.size()) {
        cerr << "Error!";
        return false;
    }

    for (int i = 0; i < _v1.size(); ++i) {
        if (abs(_v1[i] - _v2[i]) > _eps) {
            return false;
        }
    }
    return true;
}

vector<double> SolveSystem(vector<vector<double>> _s, vector<double> _b, double _eps = 1e-6) {
    vector<double> x = _b; // начальное приближение
    int n = x.size();

    int counter = 1;


    return x;
}

void Solution() {
    vector<double> vb = {pi, 5*pi/2, 51*pi/8/*, 265*pi/16, 5603*pi/128, 30075*pi/256*/};
    int n = vb.size();
    double a = 2, b = 3, h = (b - a) / (n - 1);
    vector<vector<double>> system(n, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            system[i][j] = pow(a + j*h, i);
            cout << system[i][j] << " ";
        }
        cout << "\n";
    }
    //vector<double> A = SolveSystem(system, vb);

    
}

int main() {
    Solution();

    return 0;
}