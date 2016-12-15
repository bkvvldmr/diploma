stacksize('max');
//mclose('all');
result_file = mopen("/home/vladimir/Diplom/Scilab project/result.txt", 'wt', 0);
M = 10;  // Количество тестов
for L = 2 : 30
    alfa_error = zeros(1, M); // Количество ошибок в разных тестах для одного L. В result.txt      
    for N = 1 : M
            disp("Вход в цикл");
            mkdir("/home/vladimir/Diplom/Scilab project/", "Bear_tmp"); // Копия папки Bear
            mkdir("/home/vladimir/Diplom/Scilab project/", "Woman_tmp");// Копия папки Woman
            mkdir("/home/vladimir/Diplom/Scilab project/", "Bear_Sample"); 
            mkdir("/home/vladimir/Diplom/Scilab project/", "Woman_Sample");
            Bear_list = ls("/home/vladimir/Diplom/Scilab project/Bear/*.jpg");
            Woman_list = ls("/home/vladimir/Diplom/Scilab project/Woman/*.jpg");
            for i = 1 : 120 
                copyfile(Bear_list(i), "/home/vladimir/Diplom/Scilab project/Bear_tmp");
                copyfile(Woman_list(i), "/home/vladimir/Diplom/Scilab project/Woman_tmp");
            end
            Bear_tmp_list = ls("/home/vladimir/Diplom/Scilab project/Bear_tmp/*.jpg");
            Woman_tmp_list = ls("/home/vladimir/Diplom/Scilab project/Woman_tmp/*.jpg");
            smth = 1;
            for i = 1 : L
                x_rand = int16(rand() * (120-smth)) + 1;
                y_rand = int16(rand() * (120-smth)) + 1;
                copyfile(Bear_tmp_list(x_rand), "/home/vladimir/Diplom/Scilab project/Bear_Sample");
                deletefile(Bear_tmp_list(x_rand));
                Bear_tmp_list = ls("/home/vladimir/Diplom/Scilab project/Bear_tmp/*.jpg");
                copyfile(Woman_tmp_list(y_rand), "/home/vladimir/Diplom/Scilab project/Woman_Sample");
                deletefile(Woman_tmp_list(y_rand));
                Woman_tmp_list = ls("/home/vladimir/Diplom/Scilab project/Woman_tmp/*.jpg");
                smth = smth + 1;
            end
            
            Bear_Sample_list = ls("/home/vladimir/Diplom/Scilab project/Bear_Sample/*.jpg");
            Woman_Sample_list = ls("/home/vladimir/Diplom/Scilab project/Woman_Sample/*.jpg");
            // Bear_tmp_list = ls("/home/vladimir/Diplom/Scilab project/Bear_tmp/*.jpg");
            // Woman_tmp_list = ls("/home/vladimir/Diplom/Scilab project/Woman_tmp/*.jpg");
            
            // Созданы выборки Sample из L изображений Девушек и Мишек
            // В файлах с изображениями лежат оставшиеся 120 - L изображений
            
            O_1 = cell(1, L);           // массив изображений Девушки для обучающей выборки
            O_2 = cell(1, L);           // массив изображений Мишки для обучающей выборки   
            for ii = 1 : L
                O_1(ii).entries = double(imread(Woman_Sample_list(ii)));
                O_2(ii).entries = double(imread(Bear_Sample_list(ii)));                     
            end
            //*************************************************************//
            // Теперь найдем e_1 и e_2 по формуле: e_i = 1 / L * SUM_e_i
            // где SUM_e_i = sum (j=1:L)[ f_j^i / ||f_j^i|| ]
            SUM_o_1 = zeros(300, 400, 3);
            SUM_o_2 = zeros(300, 400, 3);
            e_1 = zeros(300, 400, 3);
            e_2 = zeros(300, 400, 3);
            //w = zeros(300, 400, 3);
            for jj = 1 : L
                norm_square_o_1 = 0;
                norm_square_o_2 = 0;
                norm_square_o_1 =  sum( O_1(jj).entries(:, :, :) .* O_1(jj).entries(:, :, :) );
                norm_square_o_2 =  sum( O_2(jj).entries(:, :, :) .* O_2(jj).entries(:, :, :) );
                norm_o_1 = sqrt(norm_square_o_1);
                norm_o_2 = sqrt(norm_square_o_2);
                SUM_o_1(:, :, :) = SUM_o_1(:, :, :) + O_1(jj).entries(:, :, :) / norm_o_1;
                SUM_o_2(:, :, :) = SUM_o_2(:, :, :) + O_2(jj).entries(:, :, :) / norm_o_2;
            end
            e_1 = SUM_o_1 / L;
            e_2 = SUM_o_2 / L;
            // Получили e_1 и e_2. Далее строим w = e_2 - e_1
            w = e_2 - e_1;
            
            // Возьмем изображения из Woman_tmp_list, Bear_tmp_list
            // Если (g, w) < 0, то g = О_1. Иначе g = O_2.
            
            
            // Сначала тестируем на Девушках (O_1)
            for ll = 1 : (120 - L)
                g = double(imread(Woman_tmp_list(ll)));
                scal_mult = sum (g(:, :, :) .* w(:, :, :));
                if(scal_mult > 0)
                    alfa_error(N) = alfa_error(N) + 1;   
                end
            end
            // Теперь тестируем на Мишках (O_2)
            for ll = 1 : (120 - L)
                g = double(imread(Bear_tmp_list(ll)));
                scal_mult = sum (g(:, :, :) .* w(:, :, :));
                if(scal_mult < 0)
                    alfa_error(N) = alfa_error(N) + 1;
                end
            end
   
            
            rmdir("/home/vladimir/Diplom/Scilab project/Bear_tmp", 's');
            rmdir("/home/vladimir/Diplom/Scilab project/Woman_tmp", 's');
            rmdir("/home/vladimir/Diplom/Scilab project/Bear_Sample", 's');
            rmdir("/home/vladimir/Diplom/Scilab project/Woman_Sample", 's');
    end
    alfa_error = alfa_error / (240 - 2 * L);
    disp(L);
    mfprintf(result_file, "L: %d \n", L);
    mfprintf(result_file, "%.5f ", alfa_error(:));
    mfprintf(result_file, "\n");
end
mclose('all');





