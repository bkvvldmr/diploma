stacksize('max');
gstacksize('max');

N = 20;       // Количество векторов в линейной оболочке
M = 300 * 400;

Woman_Sample_list = ls("/home/vladimir/Diplom/Scilab project/WomanSameRakurs/*.jpg");
Sample = zeros(300, 400, N);
for n = 1 : N
    Sample(:, :, n) = rgb2gray(imread(Woman_Sample_list(n)));
end
Sample = uint8(Sample);

// Теперь создадим матрицу F = [f1 f2 ... f_N], 
//              где f_i - изображение, вытянутое в столбец
F = zeros(300 * 400, N);
F(:) = Sample(:);

[U, S, V] = svd(F, 0);

//eigen_value = zeros(1, N);   // строка сингулярных чисел
i = 1;
while S(i, i) >= S(1, 1) / 20
    eigen_value(1, i) = S(i, i);
    i = i + 1;
end 
rank_F = i - 1;  // размер базиса, если отбрасывать сингулярные числа,
                 // меньшие 10% от максимального 

E = U(:, 1 : rank_F); // базис E в "наших" обозначениях

// Создадим basis_image, который хранит "базисные" изображения 
basis_image = zeros(300, 400, rank_F);
basis_image(:) = abs(E(:));
basis_image = uint8(basis_image * 255 / max(basis_image));


























    










