function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL Р Р†Р С•Р В·Р Р†РЎР‚Р В°РЎвЂ°Р В°Р ВµРЎвЂљ РЎвЂћРЎС“Р Р…Р С”РЎвЂ Р С‘РЎР‹ Р С–Р В°РЎС“РЎРѓРЎРѓР С•Р Р†РЎРѓР С”Р С•Р С–Р С• РЎРЏР Т‘РЎР‚Р В° Р Т‘Р В»РЎРЏ Р В·Р Р…Р В°РЎвЂЎР ВµР Р…Р С‘Р в„– x1 Р С‘ x2

% Р вЂќР ВµР В»Р В°Р ВµР С РЎвЂљР В°Р С”, РЎвЂЎРЎвЂљР С•Р В±РЎвЂ№ x1 Р С‘ x2 Р В±РЎвЂ№Р В»Р С‘ РЎвЂљР С•РЎвЂЎР Р…Р С• Р Р†Р ВµР С”РЎвЂљР С•РЎР‚Р В°Р СР С‘-РЎРѓРЎвЂљР С•Р В»Р В±РЎвЂ Р В°Р СР С‘.
x1 = x1(:); x2 = x2(:);

% Р вЂ™РЎвЂ№ Р Т‘Р С•Р В»Р В¶Р Р…РЎвЂ№ Р Р†Р ВµРЎР‚Р Р…РЎС“РЎвЂљРЎРЉ Р С”Р С•РЎР‚РЎР‚Р ВµР С”РЎвЂљР Р…РЎвЂ№Р Вµ Р В·Р Р…Р В°РЎвЂЎР ВµР Р…Р С‘РЎРЏ РЎРѓР В»Р ВµР Т‘РЎС“РЎР‹РЎвЂ°Р С‘РЎвЂ¦ Р С—Р ВµРЎР‚Р ВµР СР ВµР Р…Р Р…РЎвЂ№РЎвЂ¦
sim = 0;

% ====================== Р СњР С’Р В§Р С’Р вЂєР С› Р вЂ™Р С’Р РЃР вЂўР вЂњР С› Р С™Р С›Р вЂќР С’ ======================
% Р ВР Р…РЎРѓРЎвЂљРЎР‚РЎС“Р С”РЎвЂ Р С‘Р С‘: Р СњР В°Р С—Р С‘РЎв‚¬Р С‘РЎвЂљР Вµ Р С”Р С•Р Т‘ РЎвЂћРЎС“Р Р…Р С”РЎвЂ Р С‘Р С‘, Р С”Р С•РЎвЂљР С•РЎР‚Р В°РЎРЏ Р Р†Р С•Р В·Р Р†РЎР‚Р В°РЎвЂ°Р В°Р ВµРЎвЂљ Р В·Р Р…Р В°РЎвЂЎР ВµР Р…Р С‘Р Вµ Р СР ВµРЎР‚РЎвЂ№ Р В±Р В»Р С‘Р В·Р С•РЎРѓРЎвЂљР С‘ Р Р†Р ВµР С”РЎвЂљР С•РЎР‚Р С•Р Р† x1 Р С‘ x2
%               РЎРѓ Р С—Р С•Р СР С•РЎвЂ°РЎРЉРЎР‹ Р С–Р В°РЎС“РЎРѓРЎРѓР С•Р Р†Р С”Р С•Р С–Р С• РЎРЏР Т‘РЎР‚Р В° РЎРѓ Р С—Р В°РЎР‚Р В°Р СР ВµРЎвЂљРЎР‚Р С•Р С sigma
%               
%
%

diff = x1 - x2;
squared_distance = diff' * diff;
sim = exp(-squared_distance / (2 * sigma^2));

% ====================== Р С™Р С›Р СњР вЂўР В¦ Р вЂ™Р С’Р РЃР вЂўР вЂњР С› Р С™Р С›Р вЂќР С’ ======================
    
end