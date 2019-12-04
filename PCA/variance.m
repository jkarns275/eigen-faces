function p = variance(eigvalue, th)
    vsum=sum(eigvalue);
    S = 0;
    p=0;
    while S/vsum<th
        p=p+1;
        S=S+eigvalue(p);
    end