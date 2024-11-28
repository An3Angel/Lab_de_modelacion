%Cognitive Efficiency (Gs+Gwm)
% Solución de una EDO usando métodos numéricos: Euler explícito, implícito,
% mejorado, Runge-Kutta de 4º orden y Taylor de 2º Orden

% Parámetros iniciales
t_inicial =  6.11732;       % Tiempo inicial en años
t_final = 88.15642;        % Tiempo final en años
n = 96;                   % Número de pasos
t_values = [
    6.11732, 6.28492, 6.53631, 6.70391, 6.87151, 7.12291, 7.20670, ...
    7.62570, 7.87709, 8.12849, 8.37989, 8.71508, 8.88268, 9.21788, ...
    9.63687, 9.97207, 10.55866, 10.97765, 11.48045, 11.98324, 12.40223, ...
    13.07263, 13.82682, 14.49721, 15.50279, 16.50838, 17.43017, 18.18436, ...
    19.27374, 20.27933, 21.70391, 23.12849, 24.30168, 25.47486, 26.81564, ...
    28.15642, 29.32961, 30.50279, 31.67598, 33.01676, 34.69274, 33.85475, ...
    35.78212, 36.87151, 38.04469, 39.30168, 40.47486, 41.98324, 43.24022, ...
    41.14525, 44.32961, 45.41899, 46.42458, 47.26257, 48.01676, 48.93855, ...
    49.94413, 50.94972, 51.70391, 52.62570, 53.79888, 54.72067, 55.55866, ...
    56.64804, 57.65363, 58.74302, 59.66480, 60.50279, 61.42458, 62.51397, ...
    63.60335, 64.60894, 65.53073, 66.36872, 67.29050, 68.04469, 69.05028, ...
    69.88827, 70.64246, 71.48045, 72.40223, 73.49162, 74.58101, 75.67039, ...
    76.92737, 78.01676, 79.10615, 79.94413, 80.94972, 82.03911, 83.12849, ...
    84.38547, 85.39106, 86.31285, 87.15084, 88.15642
];

y0 = 11.90808;             % Condición inicial

% Definición de la ecuación diferencial
f = @(t, y) 18.3401 - 0.167*y - 0.112*t + 0.395*sin(y) - 0.143*cos(y) + 0.801*sin(t) - 0.161*cos(t);

% ==================== Método de Euler Explícito ====================
y_explicit = zeros(1, n);   % Inicialización de las soluciones
y_explicit(1) = y0;         % Condición inicial

for i = 1:n-1
    h = t_values(i+1) - t_values(i); % Tamaño de paso
    y_explicit(i+1) = y_explicit(i) + h * f(t_values(i), y_explicit(i));
end

% ==================== Método de Euler Implícito ====================
y_implicit = zeros(1, n);   % Inicialización de las soluciones
y_implicit(1) = y0;         % Condición inicial
tol = 1e-6;                 % Tolerancia para iteración implícita
max_iter = 100;             % Máximo de iteraciones permitidas

for i = 1:n-1
    h = t_values(i+1) - t_values(i); % Tamaño de paso
    t_next = t_values(i+1);          % Tiempo en el siguiente paso
    y_guess = y_implicit(i);         % Valor inicial de iteración

    % Iteraciones de punto fijo
    for iter = 1:max_iter
        y_new = y_implicit(i) + h * f(t_next, y_guess);
        if abs(y_new - y_guess) < tol
            break;
        end
        y_guess = y_new; % Actualización del valor
    end
    y_implicit(i+1) = y_new; % Solución en el paso actual
end

% ==================== Método de Euler Mejorado ====================
y_improved = zeros(1, n);   % Inicialización de las soluciones
y_improved(1) = y0;         % Condición inicial

for i = 1:n-1
    h = t_values(i+1) - t_values(i); % Tamaño de paso
    t = t_values(i);                 % Tiempo actual
    y_pred = y_improved(i) + h * f(t, y_improved(i)); % Predicción
    y_improved(i+1) = y_improved(i) + (h/2) * (f(t, y_improved(i)) + f(t + h, y_pred)); % Corrección
end

% ==================== Método de Runge-Kutta de 4º Orden ====================
y_rk = zeros(1, n);         % Inicialización de las soluciones
y_rk(1) = y0;               % Condición inicial

for i = 1:n-1
    h = t_values(i+1) - t_values(i); % Tamaño de paso
    t = t_values(i);                 % Tiempo actual
    k1 = f(t, y_rk(i));
    k2 = f(t + h/2, y_rk(i) + h/2 * k1);
    k3 = f(t + h/2, y_rk(i) + h/2 * k2);
    k4 = f(t + h, y_rk(i) + h * k3);
    y_rk(i+1) = y_rk(i) + (h/6) * (k1 + 2*k2 + 2*k3 + k4);
end

% ==================== Método de Taylor de 2º Orden ====================
syms t y
f_sym = f; % Define f simbólicamente

% Calcula las derivadas parciales
dfdt_sym = diff(f_sym, t); % Derivada parcial respecto a t
dfdy_sym = diff(f_sym, y); % Derivada parcial respecto a y

% Convierte las expresiones simbólicas a funciones anónimas
dfdt = matlabFunction(dfdt_sym, 'Vars', [t, y]);
dfdy = matlabFunction(dfdy_sym, 'Vars', [t, y]);

% Inicialización de las soluciones para el método de Taylor de 2º Orden
y_taylor = zeros(1, n);   % Almacenar las soluciones
y_taylor(1) = y0;         % Condición inicial

% Método de Taylor de 2º Orden
for i = 1:n-1
    h = t_values(i+1) - t_values(i); % Tamaño de paso variable
    t = t_values(i);                 % Tiempo actual
    y = y_taylor(i);                 % Valor actual de y
    
    % Taylor de 2º Orden
    y_taylor(i+1) = y + h * f(t, y) + (h^2 / 2) * (dfdt(t, y) + dfdy(t, y) * f(t, y));
end

% ==================== Resultados ====================
%disp('Resultados (primeros 10 valores):');
%disp('Euler explícito:'); disp(y_explicit(1:10)');
%disp('Euler implícito:'); disp(y_implicit(1:10)');
%disp('Euler mejorado:');  disp(y_improved(1:10)');
%disp('Runge-Kutta:');     disp(y_rk(1:10)');
%disp('Método de Taylor de 2º Orden:');  disp(y_taylor(1:10)');

% ==================== Calcular el error cuadrático ====================
% Valores originales
y_original = [
    11.90808, 13.99721, 16.29526, 18.38440, 19.84680, 22.56267, 26.11421, ...
    28.41226, 31.12813, 33.84401, 36.76880, 39.48468, 42.40947, 45.33426, ...
    48.46797, 51.60167, 55.15320, 58.07799, 61.62953, 64.34540, 67.06128, ...
    69.98607, 72.91086, 75.62674, 78.55153, 80.84958, 82.31198, 83.77437, ...
    84.81894, 85.65460, 86.69916, 86.69916, 86.90808, 87.11699, 87.11699, ...
    87.11699, 87.11699, 86.90808, 86.90808, 86.69916, 86.49025, 86.49025, ...
    86.28134, 85.86351, 85.65460, 85.44568, 85.02786, 84.61003, 84.19220, ...
    85.02786, 83.77437, 83.35655, 83.14763, 82.93872, 82.52089, 82.10306, ...
    81.68524, 81.26741, 80.84958, 80.43175, 80.01393, 79.38719, 78.96936, ...
    78.34262, 77.92479, 77.08914, 76.67131, 76.04457, 75.20891, 74.58217, ...
    73.95543, 73.11978, 72.49304, 71.65738, 71.44847, 70.40390, 69.56825, ...
    68.73259, 68.10585, 67.47911, 66.43454, 65.59889, 64.76323, 63.71866, ...
    62.67409, 61.62953, 60.58496, 59.74930, 58.70474, 57.45125, 56.61560, ...
    55.77994, 54.31755, 53.48189, 52.64624, 51.81058
];

% Calcular errores cuadráticos
error_explicit = ((y_explicit - y_original).^2)/n;
error_implicit = ((y_implicit - y_original).^2)/n;
error_improved = ((y_improved - y_original).^2)/n;
error_rk = ((y_rk - y_original).^2)/n;
error_taylor = ((y_taylor - y_original).^2)/n;


% Calcular errores cuadráticos
error_explicit = ((y_explicit - y_original).^2)/n;
error_implicit = ((y_implicit - y_original).^2)/n;
error_improved = ((y_improved - y_original).^2)/n;
error_rk = ((y_rk - y_original).^2)/n;
error_taylor = ((y_taylor - y_original).^2)/n;



% ==================== Crear tablas completas ====================
tabla_euler_explicito = table(t_values', y_explicit', y_original', error_explicit', ...
    'VariableNames', {'t', 'y_explicit', 'y_original', 'error_cuadratico'});

tabla_euler_implicito = table(t_values', y_implicit', y_original', error_implicit', ...
    'VariableNames', {'t', 'y_implicit', 'y_original', 'error_cuadratico'});

tabla_euler_mejorado = table(t_values', y_improved', y_original', error_improved', ...
    'VariableNames', {'t', 'y_improved', 'y_original', 'error_cuadratico'});

tabla_runge_kutta = table(t_values', y_rk', y_original', error_rk', ...
    'VariableNames', {'t', 'y_rk', 'y_original', 'error_cuadratico'});

tabla_taylor = table(t_values', y_taylor', y_original', error_taylor', ...
     'VariableNames', {'t', 'y_taylor', 'y_original', 'error_cuadratico'});

% ==================== Escribir todas las tablas en el archivo Excel ====================
nombreArchivo = 'Polinomios + sinusoides Sol. Númerica Cognitive Efficiency (Gs+Gwm).xlsx';

writetable(tabla_euler_explicito, nombreArchivo, 'Sheet', 'Euler_Explicito');
writetable(tabla_euler_implicito, nombreArchivo, 'Sheet', 'Euler_Implicito');
writetable(tabla_euler_mejorado, nombreArchivo, 'Sheet', 'Euler_Mejorado');
writetable(tabla_runge_kutta, nombreArchivo, 'Sheet', 'Runge_Kutta');
writetable(tabla_taylor, nombreArchivo, 'Sheet', 'Taylor_2do_Orden');