format long
 
 % test part
 n1=10;
 n2=3;
% just call the primitive_roots_unity function, and it will show the image primitive n-th roots of unity
 primitive_roots_unity(n1,n2)
 w = waitforbuttonpress;
axes;
 exercise33b()
 
function exercise33b()
    a = 0:pi/30:5*pi
y1 = sin(a)
y2 = 1/(2i)*(exp(i*a)-exp(-i*a))
diff = y1-y2
subplot(3,1,1)
plot(a,y1,'g-','LineWidth',2)
xlabel('a')
ylabel('sin(a)')
title("function sin(a)")

subplot(3,1,2)
plot(a,y2,'r-','LineWidth',2)
xlabel('a')
ylabel('1/(2i)*(exp(i*a)-exp(-i*a)')
title("function 1/(2i)*(exp(i*a)-exp(-i*a)")

subplot(3,1,3)
plot(a,diff,'b-','LineWidth',2)
xlabel('a')
ylabel('Difference')
title("The difference of sin(a) and 1/(2i)*(exp(i*a)-exp(-i*a)")
end

 
 

 
function primitive_roots_unity(n1,n2)
if (n1==0)
    disp('ERROR')
    return
end
if (n2==0)
    disp('ERROR')
    return
end
list = zeros(n1,1);
index=1;
    for i=1:n1
        if gcd(i,n1)==1
            list(index)=i;
            index=index+1;
        end
    end
    subplot(2,1,1);
    title(strcat('primitive  ',num2str(n1),'-th root of unity'));
    
    hold on
    r=1;
th = 0:pi/50:2*pi;
xunit = r * cos(th) ;
yunit = r * sin(th) ;

x=-1:0.1:1;
y=x*0;

plot(x,y,'black')
plot(y,x,'black')




plot(xunit, yunit);

for i=1:index-1
    x=cos(list(i)*2*pi/n1);
    y=sin(list(i)*2*pi/n1);
    h=scatter(x,y,500,'black','filled');
    text(x+0.1,y+0.1,strcat('r',int2str(list(i)),'=', num2str(cos(list(i)*2*pi/n1),'%5.2f'),num2str(sin(list(i)*2*pi/n1),'%+5.2f'),'i'),'VerticalAlignment','top','HorizontalAlignment','left')
end
legend(h,'primitive root of unity')
subplot(2,1,2);
title(strcat('primitive  ',num2str(n2),'-th root of unity'));
list = zeros(n2,1);
index=1;
    for i=1:n2
        if gcd(i,n2)==1
            list(index)=i;
            index=index+1;
        end
    end
    
    hold on
    r=1;
th = 0:pi/50:2*pi;
xunit = r * cos(th) ;
yunit = r * sin(th) ;

x=-1:0.1:1;
y=x*0;

plot(x,y,'black')
plot(y,x,'black')




plot(xunit, yunit);

for i=1:index-1
    x=cos(list(i)*2*pi/n2);
    y=sin(list(i)*2*pi/n2);
    h=scatter(x,y,500,'black','filled');
    text(x+0.1,y+0.1,strcat('r',int2str(list(i)),'=', num2str(cos(list(i)*2*pi/n2),'%5.2f'),num2str(sin(list(i)*2*pi/n2),'%+5.2f'),'i'),'VerticalAlignment','top','HorizontalAlignment','left');
end
legend(h,'primitive root of unity');
end


