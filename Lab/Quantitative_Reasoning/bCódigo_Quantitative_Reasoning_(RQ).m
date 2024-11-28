%Quantitative Reasoning (RQ)
% Solución de una EDO usando métodos numéricos: Euler explícito, implícito,
% mejorado, Runge-Kutta de 4º orden y Taylor de 2º Orden

% Parámetros iniciales
t_inicial =   6.28492;       % Tiempo inicial en años
t_final = 88.15642;        % Tiempo final en años
n = 102;                   % Número de pasos
t_values = [
    6.28492, 6.45251, 6.62011, 6.87151, 7.03911, 7.37430, 7.62570, ...
    7.87709, 8.12849, 8.37989, 8.63128, 8.88268, 9.21788, 9.46927, ...
    9.72067, 10.13966, 10.39106, 10.72626, 11.14525, 11.48045, 11.89944, ...
    12.40223, 12.82123, 13.32402, 13.82682, 14.49721, 15.16760, 15.83799, ...
    16.50838, 17.17877, 17.84916, 18.68715, 19.44134, 20.19553, 20.94972, ...
    21.87151, 22.79330, 23.96648, 24.97207, 26.06145, 26.98324, 27.82123, ...
    28.74302, 29.83240, 31.08939, 32.17877, 33.26816, 34.18994, 35.19553, ...
    36.11732, 37.03911, 38.29609, 39.46927, 40.55866, 41.64804, 42.73743, ...
    43.57542, 44.49721, 45.41899, 46.42458, 47.68156, 48.60335, 49.69274, ...
    50.78212, 51.70391, 52.79330, 53.88268, 54.72067, 56.06145, 57.06704, ...
    58.32402, 59.16201, 60.00000, 61.00559, 62.17877, 63.10056, 64.10615, ...
    65.11173, 66.11732, 66.95531, 68.04469, 69.21788, 70.22346, 71.31285, ...
    72.31844, 73.24022, 74.16201, 75.08380, 76.08939, 76.92737, 77.68156, ...
    78.77095, 79.52514, 80.78212, 81.95531, 82.79330, 83.54749, 84.38547, ...
    85.47486, 86.39665, 87.31844, 88.15642
];



y0 = 14.83287;             % Condición inicial

% Definición de la ecuación diferencial
f = @(t, y) 16.9401 + -0.195*y + -0.090*t + -0.176*sin(t) + -0.063*cos(t);

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
    14.83287, 16.92201, 18.80223, 21.10028, 23.18942, 26.32312, 29.03900, ...
    31.33705, 33.84401, 36.35097, 38.44011, 40.73816, 42.82730, 44.91643, ...
    47.21448, 49.30362, 51.39276, 53.48189, 55.57103, 57.45125, 59.33148, ...
    61.42061, 63.30084, 65.38997, 67.06128, 68.31476, 69.56825, 70.40390, ...
    71.44847, 71.86630, 72.49304, 73.11978, 73.53760, 73.74652, 73.74652, ...
    73.74652, 73.74652, 73.74652, 73.74652, 73.53760, 73.53760, 73.32869, ...
    73.11978, 72.91086, 72.70195, 72.49304, 72.07521, 72.07521, 71.86630, ...
    71.65738, 71.44847, 71.03064, 70.61281, 70.40390, 69.98607, 69.77716, ...
    69.56825, 69.15042, 68.94150, 68.73259, 68.52368, 67.89694, 67.68802, ...
    67.27019, 66.85237, 66.64345, 66.01671, 65.80780, 65.18106, 64.97214, ...
    64.34540, 64.13649, 63.71866, 63.09192, 62.67409, 62.25627, 61.62953, ...
    61.21170, 60.79387, 60.37604, 59.74930, 59.12256, 58.49582, 57.86908, ...
    57.24234, 56.61560, 55.98886, 55.57103, 54.94429, 54.31755, 53.89972, ...
    53.27298, 52.85515, 52.01950, 51.18384, 50.76602, 50.13928, 49.51253, ...
    48.67688, 48.05014, 47.63231, 46.79666
];




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
nombreArchivo = 'Polinomios + sinusoides Sol. Númerica Quantitative Reasoning (RQ).xlsx';

writetable(tabla_euler_explicito, nombreArchivo, 'Sheet', 'Euler_Explicito');
writetable(tabla_euler_implicito, nombreArchivo, 'Sheet', 'Euler_Implicito');
writetable(tabla_euler_mejorado, nombreArchivo, 'Sheet', 'Euler_Mejorado');
writetable(tabla_runge_kutta, nombreArchivo, 'Sheet', 'Runge_Kutta');
writetable(tabla_taylor, nombreArchivo, 'Sheet', 'Taylor_2do_Orden');