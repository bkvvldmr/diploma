function F = NGrayImage2MatrixF(path, N)
    ImageList = ls(path + "/*.jpg");
    A = rgb2gray(imread(ImageList(1)));
    F = zeros(size(A, 1), size(A, 2), N);
    for n = 1 : N
        F(:, :, n) = rgb2gray(imread(ImageList(n)));
    end
endfunction
