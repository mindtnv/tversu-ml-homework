function [theta] = normalEqn(X, y)
%NORMALEQN Вычисляет вектор theta с помощью нормального уравнения

% Вы должны вернуть корректные значения следующих переменных  
theta = zeros(size(X, 2), 1);

% ====================== НАЧАЛО ВАШЕГО КОДА ======================
%

theta = (X'*X)^(-1)*X'*y

% ====================== КОНЕЦ ВАШЕГО КОДА ======================


end
