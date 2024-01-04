function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Нормализует признаки X 

% Вы должны вернуть корректные значения следующих переменных
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== НАЧАЛО ВАШЕГО КОДА ======================
% Подсказка: Вы можете найти полезными функции 'mean' и 'std'.
%       
mu = mean(X_norm);
sigma = std(X_norm);    
X_norm = X_norm - mu;
X_norm = X_norm ./ sigma;

% ====================== КОНЕЦ ВАШЕГО КОДА ======================

end
