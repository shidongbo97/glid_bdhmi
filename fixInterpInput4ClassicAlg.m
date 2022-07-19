function [x, w] = fixInterpInput4ClassicAlg(x, w)

if size(w, 2)>2
    i = find(w>1e-8);
    if numel(i)==1
        if i==1
            i = [1 2];
        else 
            i = [i-1 i];
        end
    end

    assert(numel(i)==2, 'only 2 keyframes are supported for the chosen algorithm!');
    
    x = x(i); w = w(i);
end
