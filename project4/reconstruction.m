function [construct,y]=reconstruction(input,k,eigenvector)

    PC= eigenvector(:, 1:k);
    y=PC'*input;
    construct = PC * y;
%     new_face= uint8(normalize(construct, 0, 255));
    
end