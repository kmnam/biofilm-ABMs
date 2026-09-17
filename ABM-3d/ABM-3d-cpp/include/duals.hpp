/**
 * Authors:
 *     Kee-Myoung Nam
 *
 * Last updated:
 *     7/30/2026
 */

#ifndef FORWARD_MODE_AUTODIFF_HPP
#define FORWARD_MODE_AUTODIFF_HPP

#include <iostream>
#include <cmath>
#include <concepts>
#include <array>
#include <utility>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <boost/multiprecision/mpfr.hpp>

/**
 * A lightweight class for dual numbers. 
 */
namespace Dual {

template <typename T>
class DualNumber
{
    private:
        T value;
        T derivative;  
    
    public:
        DualNumber()
        {
            this->value = 0; 
            this->derivative = 0; 
        }

        DualNumber(const T value)
        {
            this->value = value; 
            this->derivative = 0; 
        }

        DualNumber(const T value, const T derivative)
        {
            this->value = value; 
            this->derivative = derivative; 
        }

        T getValue() const
        {
            return this->value; 
        }

        T getDerivative() const 
        {
            return this->derivative; 
        }

        void setValue(const T value)
        {
            this->value = value; 
        }

        void setDerivative(const T derivative)
        {
            this->derivative = derivative; 
        }

        // ----------------------------------------------------------- // 
        //                     BINARY COMPARATORS                      //
        // ----------------------------------------------------------- // 
        bool hasSameValue(const DualNumber& other) const
        {
            return this->value == other.getValue(); 
        }

        friend bool operator==(const DualNumber& lhs, const DualNumber& rhs)
        {
            return (lhs.value == rhs.value && lhs.derivative == rhs.derivative);  
        }

        friend bool operator!=(const DualNumber& lhs, const DualNumber& rhs)
        {
            return (lhs.value != rhs.value || lhs.derivative != rhs.derivative); 
        }

        friend bool operator<(const DualNumber& lhs, const DualNumber& rhs)
        {
            return lhs.value < rhs.value; 
        } 

        friend bool operator>(const DualNumber& lhs, const DualNumber& rhs)
        {
            return lhs.value > rhs.value; 
        } 

        friend bool operator<=(const DualNumber& lhs, const DualNumber& rhs)
        {
            return lhs.value <= rhs.value; 
        }

        friend bool operator>=(const DualNumber& lhs, const DualNumber& rhs)
        {
            return lhs.value >= rhs.value; 
        }

        // ----------------------------------------------------------- // 
        //                          NEGATION                           //
        // ----------------------------------------------------------- // 
        DualNumber operator-() const 
        {
            return DualNumber(-this->value, -this->derivative); 
        }

        // ----------------------------------------------------------- // 
        //                COMPOUND ASSIGNMENT OPERATORS                //
        // ----------------------------------------------------------- // 
        DualNumber& operator+=(const DualNumber& other)
        {
            this->value += other.getValue(); 
            this->derivative += other.getDerivative();

            return *this; 
        }

        DualNumber& operator-=(const DualNumber& other)
        {
            this->value -= other.getValue(); 
            this->derivative -= other.getDerivative(); 

            return *this; 
        }

        DualNumber& operator*=(const DualNumber& other)
        {
            T curr_val = this->value; 
            T curr_deriv = this->derivative; 
            T other_val = other.getValue(); 
            T other_deriv = other.getDerivative(); 
            this->value = curr_val * other_val; 
            this->derivative = curr_val * other_deriv + curr_deriv * other_val;

            return *this;  
        }

        DualNumber& operator/=(const DualNumber& other)
        {
            T curr_val = this->value; 
            T curr_deriv = this->derivative; 
            T other_val = other.getValue(); 
            T other_deriv = other.getDerivative(); 
            this->value = curr_val / other_val; 
            this->derivative = (
                (other_val * curr_deriv - curr_val * other_deriv) /
                (other_val * other_val)
            ); 

            return *this;  
        }

        // ----------------------------------------------------------- // 
        //       COMPOUND ASSIGNMENT OPERATORS WITH PLAIN SCALARS      //
        // ----------------------------------------------------------- // 
        DualNumber& operator+=(const T& other)
        {
            this->value += other; 
            return *this;  
        }

        DualNumber& operator-=(const T& other)
        {
            this->value -= other; 
            return *this; 
        }

        DualNumber& operator*=(const T& other)
        {
            this->value *= other; 
            this->derivative *= other; 
            return *this; 
        }

        DualNumber& operator/=(const T& other)
        {
            this->value /= other; 
            this->derivative /= other; 
            return *this;
        }

        // ----------------------------------------------------------- //
        //                    MATHEMATICAL FUNCTIONS                   //
        // ----------------------------------------------------------- //
        DualNumber sin() const
        {
            using std::sin; 
            using boost::multiprecision::sin;
            using std::cos; 
            using boost::multiprecision::cos;  

            return DualNumber(
                sin(this->value),
                this->derivative * cos(this->value)
            ); 
        }

        DualNumber cos() const 
        {
            using std::sin; 
            using boost::multiprecision::sin;
            using std::cos; 
            using boost::multiprecision::cos;  

            return DualNumber(
                cos(this->value),
                -this->derivative * sin(this->value)
            ); 
        }

        DualNumber tan() const 
        {
            using std::cos; 
            using boost::multiprecision::cos;
            using std::tan;  
            using boost::multiprecision::tan;

            return DualNumber(
                tan(this->value), 
                this->derivative / (cos(this->value) * cos(this->value))
            ); 
        }

        DualNumber exp() const 
        {
            using std::exp; 
            using boost::multiprecision::exp; 

            return DualNumber(
                exp(this->value), 
                this->derivative * exp(this->value)
            ); 
        }

        DualNumber log() const 
        {
            using std::log; 
            using boost::multiprecision::log; 

            return DualNumber(
                log(this->value), 
                this->derivative / this->value
            ); 
        }

        DualNumber log10() const 
        {
            using std::log10; 
            using boost::multiprecision::log10; 

            return DualNumber(
                log10(this->value), 
                this->derivative / (this->value * log(10))
            ); 
        }

        DualNumber sqrt() const 
        {
            using std::sqrt;  
            using boost::multiprecision::sqrt;

            return DualNumber(
                sqrt(this->value), 
                0.5 * this->derivative / sqrt(this->value)
            ); 
        }

        DualNumber pow(const T& p) const 
        {
            using std::pow; 
            using boost::multiprecision::pow; 

            return DualNumber(
                pow(this->value, p), 
                this->derivative * p * pow(this->value, p - 1)
            ); 
        }
};

// ----------------------------------------------------------- // 
//            BINARY COMPARATORS WITH PLAIN SCALARS            //
// ----------------------------------------------------------- // 
template <typename T>
bool operator==(const DualNumber<T>& lhs, const T& rhs)
{
    return lhs.getValue() == rhs; 
}

template <typename T>
bool operator==(const T& lhs, const DualNumber<T>& rhs)
{
    return lhs == rhs.getValue(); 
}

template <typename T>
bool operator!=(const DualNumber<T>& lhs, const T& rhs)
{
    return lhs.getValue() != rhs; 
}

template <typename T>
bool operator!=(const T& lhs, const DualNumber<T>& rhs)
{
    return lhs != rhs.getValue(); 
}

template <typename T>
bool operator<(const DualNumber<T>& lhs, const T& rhs)
{
    return lhs.getValue() < rhs; 
}

template <typename T>
bool operator<(const T& lhs, const DualNumber<T>& rhs)
{
    return lhs < rhs.getValue(); 
}

template <typename T>
bool operator>(const DualNumber<T>& lhs, const T& rhs)
{
    return lhs.getValue() > rhs; 
}

template <typename T>
bool operator>(const T& lhs, const DualNumber<T>& rhs)
{
    return lhs > rhs.getValue(); 
}

template <typename T>
bool operator<=(const DualNumber<T>& lhs, const T& rhs)
{
    return lhs.getValue() <= rhs; 
}

template <typename T>
bool operator<=(const T& lhs, const DualNumber<T>& rhs)
{
    return lhs <= rhs.getValue(); 
}

template <typename T>
bool operator>=(const DualNumber<T>& lhs, const T& rhs)
{
    return lhs.getValue() >= rhs; 
}

template <typename T>
bool operator>=(const T& lhs, const DualNumber<T>& rhs)
{
    return lhs >= rhs.getValue(); 
}

// ----------------------------------------------------------- // 
//                       BINARY OPERATORS                      //
// ----------------------------------------------------------- // 
template <typename T>
DualNumber<T> operator+(DualNumber<T> lhs, const DualNumber<T>& rhs)
{
    lhs += rhs; 
    return lhs; 
} 

template <typename T>
DualNumber<T> operator-(DualNumber<T> lhs, const DualNumber<T>& rhs)
{
    lhs -= rhs; 
    return lhs;
}

template <typename T>
DualNumber<T> operator*(DualNumber<T> lhs, const DualNumber<T>& rhs)
{
    lhs *= rhs; 
    return lhs; 
}

template <typename T>
DualNumber<T> operator/(DualNumber<T> lhs, const DualNumber<T>& rhs)
{
    lhs /= rhs; 
    return lhs; 
}

// ----------------------------------------------------------- // 
//             BINARY OPERATORS WITH PLAIN SCALARS             //
// ----------------------------------------------------------- // 
template <typename T>
DualNumber<T> operator+(DualNumber<T> lhs, const T& rhs)
{
    lhs += rhs; 
    return lhs; 
}

template <typename T>
DualNumber<T> operator+(const T& lhs, DualNumber<T> rhs)
{
    rhs += lhs; 
    return rhs; 
}

template <typename T>
DualNumber<T> operator-(DualNumber<T> lhs, const T& rhs)
{
    lhs -= rhs; 
    return lhs; 
}

template <typename T>
DualNumber<T> operator-(const T& lhs, const DualNumber<T>& rhs)
{
    return DualNumber<T>(lhs - rhs.getValue(), -rhs.getDerivative()); 
}

template <typename T>
DualNumber<T> operator*(DualNumber<T> lhs, const T& rhs)
{
    lhs *= rhs; 
    return lhs; 
}

template <typename T>
DualNumber<T> operator*(const T& lhs, DualNumber<T> rhs)
{
    rhs *= lhs; 
    return rhs; 
}

template <typename T>
DualNumber<T> operator/(DualNumber<T> lhs, const T& rhs)
{
    lhs /= rhs; 
    return lhs; 
}

template <typename T>
DualNumber<T> operator/(const T& lhs, const DualNumber<T>& rhs)
{
    return DualNumber<T>(
        lhs / rhs.getValue(),
        -lhs * rhs.getDerivative() / (rhs.getValue() * rhs.getValue()) 
    );
}

// ----------------------------------------------------------- // 
//      BINARY OPERATORS WITH PLAIN SCALARS OF OTHER TYPES     //
// ----------------------------------------------------------- // 
template <typename T, typename U> requires std::convertible_to<U, T>
DualNumber<T> operator+(DualNumber<T> lhs, const U& rhs)
{
    lhs += rhs; 
    return lhs; 
}

template <typename T, typename U> requires std::convertible_to<U, T>
DualNumber<T> operator+(const U& lhs, DualNumber<T> rhs)
{
    rhs += lhs; 
    return rhs; 
}

template <typename T, typename U> requires std::convertible_to<U, T>
DualNumber<T> operator-(DualNumber<T> lhs, const U& rhs)
{
    lhs -= rhs; 
    return lhs; 
}

template <typename T, typename U> requires std::convertible_to<U, T>
DualNumber<T> operator-(const U& lhs, const DualNumber<T>& rhs)
{
    return DualNumber<T>(lhs - rhs.getValue(), -rhs.getDerivative()); 
}

template <typename T, typename U> requires std::convertible_to<U, T>
DualNumber<T> operator*(DualNumber<T> lhs, const U& rhs)
{
    lhs *= rhs; 
    return lhs; 
}

template <typename T, typename U> requires std::convertible_to<U, T>
DualNumber<T> operator*(const U& lhs, DualNumber<T> rhs)
{
    rhs *= lhs; 
    return rhs; 
}

template <typename T, typename U> requires std::convertible_to<U, T>
DualNumber<T> operator/(DualNumber<T> lhs, const U& rhs)
{
    lhs /= rhs; 
    return lhs; 
}

template <typename T, typename U> requires std::convertible_to<U, T>
DualNumber<T> operator/(const U& lhs, const DualNumber<T>& rhs)
{
    return DualNumber<T>(
        lhs / rhs.getValue(),
        -lhs * rhs.getDerivative() / (rhs.getValue() * rhs.getValue()) 
    );
}

// ----------------------------------------------------------- // 
//             MATHEMATICAL FUNCTIONS (NON-METHOD)             //
// ----------------------------------------------------------- // 
template <typename T>
DualNumber<T> sin(const DualNumber<T>& x)
{
    return x.sin(); 
}

template <typename T>
DualNumber<T> cos(const DualNumber<T>& x)
{
    return x.cos(); 
}

template <typename T>
DualNumber<T> tan(const DualNumber<T>& x)
{
    return x.tan(); 
}

template <typename T>
DualNumber<T> exp(const DualNumber<T>& x)
{
    return x.exp(); 
}

template <typename T>
DualNumber<T> log(const DualNumber<T>& x)
{
    return x.log(); 
}

template <typename T>
DualNumber<T> log10(const DualNumber<T>& x)
{
    return x.log10(); 
}

template <typename T>
DualNumber<T> sqrt(const DualNumber<T>& x)
{
    return x.sqrt(); 
}

template <typename T>
DualNumber<T> pow(const DualNumber<T>& x, const T& p)
{
    return x.pow(p); 
}

template <typename T, typename U> requires std::convertible_to<U, T>
DualNumber<T> pow(const DualNumber<T>& x, const U& p)
{
    return x.pow(p); 
}

// ----------------------------------------------------------- // 
//    MORE MATHEMATICAL FUNCTIONS (FOR EIGEN COMPATIBILITY)    //
// ----------------------------------------------------------- // 
template <typename T>
DualNumber<T> conj(const DualNumber<T>& x)
{
    return x; 
}

template <typename T>
DualNumber<T> real(const DualNumber<T>& x)
{
    return x; 
}

template <typename T>
DualNumber<T> imag(const DualNumber<T>& x)
{
    return DualNumber<T>(0); 
}

template <typename T>
DualNumber<T> abs2(const DualNumber<T>& x)
{
    return x * x; 
}

// ----------------------------------------------------------- //
//                       OTHER FUNCTIONS                       //
// ----------------------------------------------------------- //
template <typename T>
std::ostream& operator<<(std::ostream& os, const DualNumber<T>& x)
{
    os << "Dual(" << x.getValue() << ", " << x.getDerivative() << ")"; 

    return os; 
}

}   // namespace Dual

// ----------------------------------------------------------- // 
//      NUMTRAITS SPECIALIZATION FOR EIGEN COMPATIBILITY       //
// ----------------------------------------------------------- // 
#include <Eigen/Core>

namespace Eigen {

template <typename T>
struct NumTraits<Dual::DualNumber<T> > : NumTraits<T>
{
    using DualType = Dual::DualNumber<T>; 

    using Real = DualType; 
    using NonInteger = DualType; 
    using Nested = DualType; 
    using Literal = typename NumTraits<T>::Literal; 

    enum
    {
        IsComplex = 0, 
        IsInteger = 0, 
        IsSigned = NumTraits<T>::IsSigned, 
        RequireInitialization = 1, 
        ReadCost = 2 * NumTraits<T>::ReadCost, 
        AddCost = 2 * NumTraits<T>::AddCost,
        MulCost = 3 * NumTraits<T>::MulCost + NumTraits<T>::AddCost
    }; 
}; 

}   // namespace Eigen

// ----------------------------------------------------------- // 
//                     GRADIENT CALCULATION                    // 
// ----------------------------------------------------------- // 
namespace Dual {

using namespace Eigen; 

// Vectors and matrices of dual numbers
template <typename T>
using DualVector = Matrix<DualNumber<T>, Dynamic, 1>; 

template <typename T>
using DualMatrix = Matrix<DualNumber<T>, Dynamic, Dynamic>; 

// Scalar-valued functions 
template <typename T>
using ScalarFunc = std::function<DualNumber<T>(const Ref<const DualVector<T> >&)>; 

template <typename T>
Matrix<T, Dynamic, 1> gradient(ScalarFunc<T>& func,
                               const Ref<const Matrix<T, Dynamic, 1> >& x)
{
    const int n = x.size(); 
    Matrix<T, Dynamic, 1> grad(n); 

    // Calculate the partial derivative w.r.t each variable
    for (int i = 0; i < n; ++i)
    {
        // Prepare a vector of dual numbers 
        Matrix<DualNumber<T>, Dynamic, 1> x_in(n);
        for (int j = 0; j < n; ++j)
        {
            T deriv = (j == i ? 1 : 0); 
            x_in(j) = DualNumber<T>(x(j), deriv); 
        } 

        // Evaluate the function ... 
        DualNumber<T> f = func(x_in);

        // ... and extract the partial derivative  
        grad(i) = f.getDerivative(); 
    }

    return grad;  
}

// Vector-valued functions 
template <typename T>
using VectorFunc = std::function<DualVector<T>(const Ref<const DualVector<T> >&)>;

template <typename T>
Matrix<T, Dynamic, Dynamic> jacobian(VectorFunc<T>& func, 
                                     const Ref<const Matrix<T, Dynamic, 1> >& x)
{
    const int n = x.size();

    // Evaluate the function once to get the output dimension 
    Matrix<DualNumber<T>, Dynamic, 1> x0(n); 
    for (int i = 0; i < n; ++i)
        x0(i) = DualNumber<T>(x(i), 0); 
    Matrix<DualNumber<T>, Dynamic, 1> f0 = func(x0); 
    const int m = f0.size(); 

    // Initialize the Jacobian matrix
    Matrix<T, Dynamic, Dynamic> jac(m, n); 

    // For each function component ... 
    for (int i = 0; i < m; ++i)
    {
        // For each variable ... 
        for (int j = 0; j < n; ++j)
        {
            // Prepare a vector of dual numbers 
            Matrix<DualNumber<T>, Dynamic, 1> x_in(n); 
            for (int k = 0; k < n; ++k)
            {
                T deriv = (k == j ? 1 : 0); 
                x_in(k) = DualNumber<T>(x(k), deriv); 
            } 

            // Evaluate the function ... 
            DualVector<T> f = func(x_in); 
            
            // ... and extract the j-th partial derivative of the i-th function
            jac(i, j) = f(i).getDerivative(); 
        }
    }

    return jac; 
}

}   // namespace Dual

#endif
