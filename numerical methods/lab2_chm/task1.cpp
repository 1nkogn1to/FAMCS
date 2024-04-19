#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace std;

int count = 1;

double F(double _x) {
    return _x * exp(-_x);
}

vector<double> Fv(vector<double> _x) {
    double _size = _x.size();
    vector<double> _result(_size);

    for (int i = 0; i < _size; ++i) {
        _result[i] = F(_x[i]);
    }
    return _result;
}

vector<vector<double>> SeparatedDifferences(vector<double> _x) {
    int _size = _x.size();

    if (_size == 0) { vector<vector<double>>(0); }
    if (_size == 1) { vector<vector<double>>(1, vector<double>(1, F(_x[0]))); }

    vector<vector<double>> _matrixResult(_size);

    _matrixResult[0] = Fv(_x);
    
    for (int i = 1; i < _size; ++i) {
        double _sizeOfColumn = _size - i;
        _matrixResult[i] = vector<double>(_sizeOfColumn);
        
        for (int j = 1; j <= _sizeOfColumn; ++j) {
            _matrixResult[i][j - 1] = (_matrixResult[i - 1][j] - _matrixResult[i - 1][j - 1]) / (_x[j - 1 + i] - _x[j - 1]);
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
        _polinom1 = _polinom2;
        _polinom2.clear();
        return;
    }

    if (_size2 == 0) {
        _polinom2.clear();
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

vector<double> Pv(vector<double> _x, vector<double> _polinom) {
    int _sizeX = _x.size();
    int _sizeP = _polinom.size();

    vector<double> _result(_sizeX, 0);
    
    for (int i = 0; i < _sizeX; ++i) {
        for (int j = 0; j < _sizeP; ++j) {
            _result[i] += _polinom[j] * pow(_x[i], j);
        }
    }

    return _result;
}

// Проверка является ли текущий полином достаточным для интерполяции
// исходной функции с точностью до 4 знака после запятой
bool Check(vector<double> _x, vector<double> _polinom, double _eps) {
    vector<double> _fValues = Fv(_x);
    vector<double> _iValues = Pv(_x, _polinom);

    int _sizeF = _fValues.size();
    int _sizeI = _iValues.size();

    if (_sizeF != _sizeI) {
        cerr << "Something went wrnog.";
        MyClear(_fValues, _iValues);
        return false;
    }

    for (int i = 0; i < _sizeF; ++i) {
        if (abs(_fValues[i] - _iValues[i]) >= _eps) {
            MyClear(_fValues, _iValues);
            return false;
        }
    }

    cout << count++ << " table:\n";
    for (int i = 0; i < _sizeF; ++i) {
        cout << setprecision(4) << _fValues[i] << " ";
    }
    cout << "\n";
    for (int i = 0; i < _sizeF; ++i) {
        cout << setprecision(4) << _iValues[i] << " ";
    }
    cout << "\n---------------------------------------------------------------------------------------------------------------------------------------------\n";

    MyClear(_fValues, _iValues);
    return true;
}

void print(vector<double> _polinom) {
    cout << "Polinom:\n";
    cout << _polinom[1] << "*x";
    for (int i = 2; i < _polinom.size(); ++i) {
        if (_polinom[i] != 0) {
            if (_polinom[i] < 0) cout << _polinom[i] << "*x^" << i;
            else cout << "+" << _polinom[i] << "*x^" << i;
        }
    }
    cout << "\n";
}

void Solution() {
    double a = 0, b = 2, h = 0.1, eps = 1e-4;
    int _size = int((b - a) / h) + 1;

    vector<double> _x(_size);
    for (int i = 0; i < _size; ++i) { _x[i] = i * h; }

    vector<vector<double>> _differences = SeparatedDifferences(_x);

    vector<double> _polinom(_size, 0);
    _polinom[0] += _differences[0][0];

    // основной цикл
    for (int i = 1; i < _size; ++i) {
        vector<double> _tempSolution = {_x[0], 1};

        for (int j = 1; j < i; ++j) {
            vector<double> _temp = {-_x[j], 1};
            PolinomialMultiplication(_tempSolution, _temp);
        }
        Polinom_x_Scalar(_tempSolution, _differences[i][0]);
        PolinomialAddition(_polinom, _tempSolution);
        
        if (Check(_x, _polinom, eps)) break;
    }
    
    // изменяем вектор иксов
    h = h * 0.5;
    int _newSize = int((b - a) / h) + 1;
    _x.resize(_newSize);
    for (int i = 0; i < _newSize; ++i) { _x[i] = i * h; }

    // используем ту же функцию для проверки погрешности, что и в основном цикле
    cout << (Check(_x, _polinom, eps) ? "All is ok, good interpolation" : "All is not ok, bad interpolation") << "\n";

    print(_polinom);

    MyClear(_polinom, _x);
    for (int i = 0; i < _size; ++i) { _differences[i].clear(); }
    _differences.clear();
}

int main() {
    Solution();

    return 0;
}