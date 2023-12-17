# Error-Propagation

This Scripts is capable of calculating the *Gaussian Error Propagation* and create simple statistics for datasets.

This was created out of the frustration with the module *Fehlerrechnung* (engl. Error-Propagation) at my university and the ability to reliably cross check my results.

### INFORMATION FOR END USERS 

- ### Math functions 
(all trigonometric functions currently only support arguments in `radians`)

argument `x` is of type ` Value | number_as_string | number`. All following functions return a `Value` instance

  - `exp( x )`
  - `log( argument [, base:x defaults to Euler-e ] )`
  - `sin( x )`
  - `cos( x )`
  - `tan( x )`
  - `arcsin( x )`
  - `arccos( x )`
  - `arctan( x )`
  - `sinh( x )`
  - `cosh( x )`
  - `tanh( x )`
  - `arsinh( x )`
  - `arcosh( x )`
  - `artanh( x )`


Following classes and functions that are relevant for you:

- ### Classes
  - ### Value
    - ```py
        Value(
            best_value: number_as_str|number,
            error: number_as_str|number
            [, scale_exponent: int=0 ]
            [, precision: int=2 ]
            [, id:str ] 
        )
        ```
    -   | operation        | symbol | example              |
        | :--------------: | :----: | -------------------- |
        | negate           | `-`    | `-a`                 |
        | positive         | `+`    | `+a`                 |
        | addition         | `+`    | `a+b` `1+a` `a+1`    |
        | subtraction      | `-`    | `a-b` `2-a` `a-2`    |
        | multiplication   | `*`    | `a*b` `3*a` `a*3`    |
        | division         | `/`    | `a/b` `4/a` `a/4`    |
        | exponentiation   | `**`   | `a**b` `5**a` `a**5` |
        | Compare Equal    | `==`   | `a==b` `6==a` `a==6` |
        | Compare Unequal  | `!=`   | `a!=b` `7!=a` `a!=7` |


  - ### Measurement
    - ```py
      Measurement( 
          data: Sequence_of_numbers
          [, scale_exponent: int=0]
          [, precision: int=2 ]
          [, id:str ]
      )
      ```
  - ### Linear Regression
    The following classes assume that *only* the `Y`-Component of the (`X`, `Y`)-Point-cloud is *erroneous*.

    - simple linear Regression (all `Y`-Values have the same absolute error)
      #### TODO: Insert here Regression_linear_simple 

    - *general* linear Regression (`Y`-Values can have different absolute errors)
      #### TODO: Insert here Regression_linear

  - ### PF [optional for advanced use]
    Converting data from one *unit-scale* to an other specific *unit-scale*.

    *PF* employs *Decimal-Prefixes* or *Pre-factors* to denote the magnitude or scale of units( i.e *kilometers* (*km*), *microfarad* (*µF*) etc ).
    
    example:
    ```py
    >>> # creating a mass `Value` in units of milligram that are converted to kilograms
    >>> Value( '100', '3.5', exp=PF.m.to(PF.k) )
    (100 ± 3.5)e-6    δe= 0.035
    
    >>> # creating a distance `Value` in units of kilometers that are converted to meters
    >>> Value( '3.21', '0.11', exp=PF.k.to(PF.NONE) )
    (3.21 ± 0.11)e+3    δe= 0.034
    ```

    - predefined class-variables

        | name    | alias | factor |
        | ------- | ----- | ------ |
        | `PETA`  | `P`   | `E+15` |
        | `TERA`  | `T`   | `E+12` |
        | `GIGA`  | `G`   | `E+9 ` |
        | `MEGA`  | `M`   | `E+6 ` |
        | `KILO`  | `k`   | `E+3 ` |
        | `HECTO` | `h`   | `E+2 ` |
        | `DECA`  | `da`  | `E+1 ` |
        | `NONE`  | `_`   | `1   ` |
        | `DEZI`  | `d`   | `E-1 ` |
        | `CENTI` | `c`   | `E-2 ` |
        | `MILLI` | `m`   | `E-3 ` |
        | `MICRO` | `µ`   | `E-6 ` |
        | `NANO`  | `n`   | `E-9 ` |
        | `PICO`  | `p`   | `E-12` |
        | `FEMTO` | `f`   | `E-15` |

### Example Code
```py
>>> # creating statistics to a set of measurements, the supplied data is scaled by the Pre-factor `PF.m` same as `exp=-3`
>>> length = Measurement( [1.249, 1.234, 1.252, 1.238, 1.235, 1.246, 1.262, 1.255, 1.243], PF.m, id='d' ) # data is in millimeter
Measurement:
+----------------------------------------------------+
|         N:              9                          |
| Spanwidth:        0.028 * 10^-3                    |
| -------------------------------------------------- |
|   Average:       (1246 ± 3.2)e-6       δe= 0.0025  |
|    Median:        (1.2 ± 0)e-3         δe= 0E+27   |
| -------------------------------------------------- |
|        σ²:        (80 ± 4.4)e-9        δe= 0.056   |
|         σ:       (8.9 ± 2.1)e-6        δe= 0.24    |
|        s²:        (90 ± 5.6)e-9        δe= 0.062   |
|         s:       (9.5 ± 2.4)e-6        δe= 0.25    |
+----------------------------------------------------+

>>> # defining erroneous values
>>> # id is optional and does not alter results of your calculations
>>> # it is mainly there to be a placeholder in the .print_info_equation() output (example is below)
>>> a = Value( '1.630' , '0.021' , prec=2, id='a' )
>>> b = Value( '0.6649', '0.0040', prec=3, id='b' )
>>> x = Value( '1.376' , '0.037' , prec=2, id='x' )

>>> a.v # alias for a.value
1.63

>>> a.error # alias for a.e
0.021

>>> a.re # alias for a.relative_error
0.013

>>> # examples displaying the syntax of different calculations
>>> y1 = 13*a*x + 14*a*b*x**2 + 21*a*b**3
67.947 ± 2.513    δe= 0.03699

>>> y2 = exp( ( a - x ) / x )
(1202.73 ± 42.48)e-3    δe= 0.03532

>>> y3 = b * sin( a * x )
(520.3 ± 27.86)e-3    δe= 0.05355

>>> y4 = ( x - a ) / ( x + b )
(-124.45 ± 22.84)e-3    δe= 0.1835

>>> y1.print_info_equation()
Equation   : b**3 * a * 21 + x * a * 13 + x**2 * b * a * 14
Derivatives:
        x:      a * 13 + b * a * x * 28
        b:      a * b**2 * 63 + x**2 * a * 14
        a:      b**3 * 21 + x * 13 + x**2 * b * 14
```