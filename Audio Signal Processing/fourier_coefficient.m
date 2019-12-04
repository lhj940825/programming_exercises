

global y_Fourier;
init()

% Part b
for k=1:16
    draw_and_calculate(k)

end


% Part c
% Test for k=5,25,50
% list=[5,25,50];
% for i=1:3
%     k=list(i);
%     draw_and_calculate(k)
% end

subplot(4,1,1);
plot(x,y_Fourier)



function init()
x_1=0:0.01:0.5;
x_2=0.5:0.01:1;

y_1=x_1*0+1;
y_2=x_2*0-1;

subplot(4,1,1);

plot(x_1,y_1,'b');
hold on
title('image of f(t) and sum of fourier series');
plot(x_2,y_2,'b');
global y_Fourier;
y_Fourier=0;
end

function draw_and_calculate(k)

integral_basis1_1 = integral(@(x)x*0+1,0,0.5);
integral_basis1_2 = integral(@(x)x*0-1,0.5,1);
para_basis_1=integral_basis1_1+integral_basis1_2;


integral_basisA_1=sqrt(2)*integral(@(x) cos(x*2*pi*k)*1,0,0.5);
integral_basisA_2=sqrt(2)*integral(@(x) cos(x*2*pi*k) * -1,0.5,1);
integral_basisA=integral_basisA_1+integral_basisA_2;
if abs(integral_basisA)<10^-10
    integral_basisA=0;
end
integral_basisB_1=sqrt(2)*integral(@(x) sin(x*2*pi*k)*1,0,0.5);
integral_basisB_2=sqrt(2)*integral(@(x) sin(x*2*pi*k) * -1,0.5,1);
integral_basisB=integral_basisB_1+integral_basisB_2;
if abs(integral_basisB)<10^-10
    integral_basisB=0;
end

x = linspace(0, 1);
y_basisA= integral_basisA*cos(x*2*pi*k);


y_basisB=integral_basisB*sin(x*2*pi*k);

y_basis1=x*0+para_basis_1;

subplot(4,1,2);
title('Fourier coefficients for basis 1')
plot(x,y_basis1)
hold on
subplot(4,1,3);
title('Fourier coefficients for basis Ak')
plot(x,y_basisA)
hold on
subplot(4,1,4);
title('Fourier coefficients for basis Bk')
plot(x,y_basisB)
hold on


global y_Fourier
y_Fourier =y_Fourier+y_basis1+sqrt(2)*y_basisA+sqrt(2)*y_basisB;

pause(.1)

end

