// tol -- доля от максимального сингулярного числа, которой пренебрегаем
function E = imOrth(I, tol)     
    N = size(I, 3);
    M = size(I, 1) * size(I, 2);
    F = zeros(M, N);
    F(:) = I(:);
    [U, S, V] = svd(F, 0);  
    rank_F = sum(S(1 : size(S, 1) + 1 : $) >= S(1) * tol);
    part_U = U(:, 1 : rank_F); 
    E = zeros(size(I, 1), size(I, 2), rank_F);
    E(:) = part_U(:);
endfunction

