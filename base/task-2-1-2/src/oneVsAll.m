function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL обучает несколько логистических регрессии и возвращает результат
%в матрице all_theta, в которой i-тая строка соответствует весам
%классификатора для класса i

% Инициализация полезных переменных
m = size(X, 1);
n = size(X, 2);

% Вы должны вернуть корректные значения следующих переменных 
all_theta = zeros(num_labels, n + 1);

% Добавляем столбец 1 в матрицу с данными X
X = [ones(m, 1) X];

% ====================== НАЧАЛО ВАШЕГО КОДА ======================
% Подсказка: theta(:) "развернет" матрицу theta в вектор-солбец.
%
% Подсказка: Возможно вы захотите использовать операцию y == c, где y - это
%       вектор, а c - скаляр.
%
% Note: В этом задании рекомендуем использовать решатель fmincg для оптимизации
%       функции стоимости. Вы также можете использовать цикл for
%       (for c = 1:num_labels) для обучения отдельных классификаторов.
%
%       fmincg работает аналогично fminunc, только он более эффективен
%       для функций с большим количеством параметров.
%
% Пример кода для fmincg:
%
%     % Инициализируем theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Задаем массив опций для fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Вызываем fmincg и получаем оптимальные значения theta
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%





% ====================== КОНЕЦ ВАШЕГО КОДА ======================


end