function z = basic_expension(x, P)
z=[];
for p=0:P

    z_temp=x.^p;
    z=horzcat(z,z_temp);

end