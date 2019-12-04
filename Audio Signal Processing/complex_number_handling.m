 format long
 
 % test part
 fir_CN = -3.3 + 0.1i; % the first complex number
 sec_CN = 2.3-3i; % the second complex number
 [p,q]= calculate_and_plot(fir_CN, sec_CN); % p should be the product and q should be the quotient
 disp("product is "+p); % output the product
 disp("quotient is"+q); % output the quotinet
 
function draw(complex, linespec)
    if real(complex)~=0
        x=0.00:real(complex)/10:real(complex);
        y=x*imag(complex)/real(complex);
        plot(x,y,linespec)
    else  % if the real part is zero, it should be a line segment of Y-axis.
        y=0.00:imag(complex)/10:imag(complex);
        x=y*0;
        plot(x,y,linespec);
    end 
end

 function [product, quotient]=calculate_and_plot(fir_CN,sec_CN) 
    product = fir_CN*sec_CN; % the product of two complex numbers
    if sec_CN~=0
        quotient = fir_CN/sec_CN; % the quotient of two complex numbers
    else
        quotient=0;   % defualt value, in case of the program crash,
        disp("ERROR") % the divisor should not be zero
    end

    subplot(2,1,1);
    draw(fir_CN,'b');
    hold on;
    draw(sec_CN,'r');
    draw(product,'g');
    title('Subplot 1: Product');
    legend('Z_1','Z_2','product(Z_1,Z_2)'); % Z : complex nunmber


    subplot(2,1,2);
    draw(fir_CN, 'b');
    hold on;
    draw(sec_CN, 'r');
    draw(quotient,'g');
    title('Subplot 2: division');
    legend('Z_1','Z_2','quotient(Z_1,Z_2)');
end

