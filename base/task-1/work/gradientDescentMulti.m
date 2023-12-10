function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI ���������� ����������� ����� ��� ������ ���������� theta

% ������������� �������� ����������
m = length(y); % ���������� ��������� ��������
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== ������ ������ ���� ======================
    % Instructions: ��������� ���� ��� ������������ ������ �� �������
    %               ���������� theta. 
    %
    % Hint: �� ����� ������� ����� ������� �������� �������� �������
    %       ��������� (computeCost) � ���������.
    %







    % ====================== ����� ������ ���� ======================

    % ��������� �������� J �� ������ ��������
    J_history(iter) = computeCost(X, y, theta);

end

end
