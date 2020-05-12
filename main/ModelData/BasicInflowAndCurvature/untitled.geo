//+
Point(1) = {0, 0, 0};
//+
Point(2) = {100, 0, 0};
//+
Point(3) = {100, 10, 0};
//+
Point(4) = {0, 10, 0};
//+
DefineConstant[curving_param = 20.0];
DefineConstant[upper_bezier_point = 10.0 + curving_param];
DefineConstant[lower_bezier_point = 0.0 - curving_param];
//+
Point(5) = {50, upper_bezier_point, 0};
//+
Point(6) = {50, lower_bezier_point, 0};
//+
Bezier(1) = {4, 5, 3};
//+
Bezier(2) = {1, 6, 2};
//+
Line(3) = {4, 1};
//+
Line(4) = {3, 2};
//+
Curve Loop(1) = {1, 4, -2, -3};
//+
Plane Surface(1) = {1};
//+
Physical Curve(1) = {1};
//+
Physical Curve(2) = {2};
//+
Physical Curve(3) = {3};
//+
Physical Curve(4) = {4};
//+
Physical Surface(5) = {1};
//+
Field[1] = Box;
Field[1].VIn = 1.0;
Field[1].VOut = 1.0;
Field[1].XMin = -1.0;
Field[1].XMax = 101.0;
Field[1].YMin = -1.0;
Field[1].YMax = 11.0;
//+
Background Field = 1;