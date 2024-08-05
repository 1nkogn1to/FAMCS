#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

double F1(double _x) {
    return _x * cos(_x + 5);
}

double F2(double _x) {
    return 1 / (1 + 25 * _x * _x);
}

vector<double> Fv(vector<double> _x, int _code) {
    double _size = _x.size();
    vector<double> _result(_size);

    switch (_code) {
        case 1:
            for (int i = 0; i < _size; ++i) {
                _result[i] = F1(_x[i]);
            }
            break;
        case 2:
            for (int i = 0; i < _size; ++i) {
                _result[i] = F2(_x[i]);
            }
            break;
        default:
            break;
    }
    return _result;    
}

long long factorial(int num) {
    long long arr[] = {1,1,2,6,24,120, 720, 5040, 40320, 362880, 3628800, 39916800, 479001600, 6227020800, 87178291200, 1307674368000};
    return arr[num];
}

vector<vector<double>> SeparatedDifferences(vector<double> _x, int _code = 2) {
    int _size = _x.size();

    if (_size == 0) { vector<vector<double>>(0); }
    if (_size == 1 && _code == 1) { vector<vector<double>>(1, vector<double>(1, F1(_x[0]))); }
    if (_size == 1 && _code == 2) { vector<vector<double>>(1, vector<double>(1, F2(_x[0]))); }

    vector<vector<double>> _matrixResult(_size);

    _matrixResult[0] = Fv(_x , _code);
    
    for (int i = 1; i < _size; ++i) {
        double _sizeOfColumn = _size - i;
        _matrixResult[i] = vector<double>(_sizeOfColumn);
        
        for (int j = 1; j <= _sizeOfColumn; ++j) {
            _matrixResult[i][j - 1] = (_matrixResult[i - 1][j] - _matrixResult[i - 1][j - 1]) / (_x[j - 1 + i] - _x[j - 1]);
            //_matrixResult[i][j - 1] /= factorial(i);
        }
    }

    return _matrixResult;
}

void MyClear(vector<double>& _v1, vector<double>& _v2) {
    _v1.clear();
    _v2.clear();
}

void PolinomialMultiplication(vector<double>& _polinom1, vector<double>& _polinom2) {
    int _size1 = _polinom1.size(), _size2 = _polinom2.size();
    
    if (_size1 == 0) {
        _polinom2.clear();
        return;
    }

    if (_size2 == 0) {
        _polinom1.clear();
        return;
    }

    vector<double> _temp = _polinom1;
    _polinom1.clear();
    _polinom1.resize(_size1 + _size2 - 1, 0);
    
    for (int i = 0; i < _size1; ++i) {
        for (int j = 0; j < _size2; ++j) {
            _polinom1[i + j] += _temp[i] * _polinom2[j];
        }
    }
    MyClear(_temp, _polinom2);
}

void Polinom_x_Scalar(vector<double>& _polinom, double _scalar) {
    int _size = _polinom.size();
    for (int i = 0; i < _size; ++i) {
        _polinom[i] *= _scalar;
    }
}

void PolinomialAddition(vector<double>& _polinom1, vector<double>& _polinom2) {
    int _size1 = _polinom1.size(), _size2 = _polinom2.size();

    if (_size1 < _size2) {
        swap(_polinom1, _polinom2);
        swap(_size1, _size2);
    }
    
    for (int i = 0; i < _size2; ++i) {
        _polinom1[i] += _polinom2[i];
    }

    _polinom2.clear();
}


void print(vector<double> _polinom) {
    cout << "Polinom:\n";
    cout << _polinom[0];
    if (_polinom[1] < 0) {
        cout << _polinom[1] << "*x";
    } else {
        cout << "+" << _polinom[1] << "*x";
    }
    for (int i = 2; i < _polinom.size(); ++i) {
        if (_polinom[i] != 0) {
            if (_polinom[i] < 0) cout << _polinom[i] << "*x^" << i;
            else cout << "+" << _polinom[i] << "*x^" << i;
        }
    }
    cout << "\n";
}


void Solution() {
    double a = -5, b = 5;
    
    vector<int> _ns = {3, 5, 7, 10, 15, 20};
    int _sizeOfNs = _ns.size();

    for (int n = 0; n < _sizeOfNs; ++n) {
        int _sizeOfCurrentX = _ns[n];
        vector<double> _x(_sizeOfCurrentX);

        double h = (b - a) / (_sizeOfCurrentX - 1);
        
        // заполняем вектор иксов
        for (int i = 0; i < _sizeOfCurrentX; ++i) { _x[i] = a + i * h; }

        // код из прошлой таски
        vector<vector<double>> _differences = SeparatedDifferences(_x);

        vector<double> _polinom(_sizeOfCurrentX, 0);
        _polinom[0] += _differences[0][0];

        // основной цикл
        for (int i = 1; i < _sizeOfCurrentX; ++i) {
            vector<double> _tempSolution = {_x[0], 1};

            for (int j = 1; j < i; ++j) {
                vector<double> _temp = {-_x[j], 1};
                PolinomialMultiplication(_tempSolution, _temp);
            }
            Polinom_x_Scalar(_tempSolution, _differences[i][0]);
            PolinomialAddition(_polinom, _tempSolution);
        }
        print(_polinom);
        
        MyClear(_polinom, _x);
        for (int i = 0; i < _sizeOfCurrentX; ++i) { _differences[i].clear(); }
        _differences.clear();
    }
}

int main() {
    Solution();

    return 0;
}