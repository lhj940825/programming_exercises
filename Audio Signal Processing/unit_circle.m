p=[1,2,3,4,10];

draw_unit_circle(p)


function draw_unit_circle(a)
 for i=1: length(a)
     x=-1:0.01:1;
        
        x_power=abs(power(x,a(i)));
        y_power=1-x_power;
        y=abs(power(y_power,1/a(i)));
        y_minus=-y;
        
    c=rand(1,3);
  h(i)=plot(x,y,'color',c);
  names(i)={num2str(a(i))};
  
  hold on;
  h1=plot(x,y_minus,'color',c);
  
 end
 
 legend(h,names)
end