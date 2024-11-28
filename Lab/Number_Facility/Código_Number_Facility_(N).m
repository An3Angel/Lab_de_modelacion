%Number Facility (N)
% Solución de una EDO usando métodos numéricos: Euler explícito, implícito,
% mejorado, Runge-Kutta de 4º orden y Taylor de 2º Orden

% Parámetros iniciales
t_inicial =   6.20112;       % Tiempo inicial en años
t_final = 72.65363;        % Tiempo final en años
n = 104;                   % Número de pasos
t_values = [
    6.20112, 6.45251, 6.53631, 6.70391, 6.95531, 7.12291, 7.29050, ...
    7.54190, 7.79330, 7.96089, 8.21229, 8.46369, 8.71508, 9.05028, ...
    9.13408, 9.46927, 9.72067, 10.05587, 10.39106, 10.72626, 11.22905, ...
    11.48045, 11.98324, 12.40223, 12.90503, 13.32402, 13.91061, 14.41341, ...
    14.91620, 15.58659, 16.42458, 17.43017, 18.35196, 19.60894, 20.69832, ...
    22.20670, 23.12849, 24.30168, 25.47486, 26.56425, 27.73743, 28.74302, ...
    29.74860, 30.50279, 31.42458, 32.43017, 33.26816, 34.10615, 35.36313, ...
    36.20112, 37.45810, 38.46369, 39.55307, 40.64246, 41.64804, 42.56983, ...
    43.57542, 44.58101, 45.50279, 46.59218, 47.84916, 48.85475, 49.77654, ...
    51.03352, 52.12291, 53.21229, 54.30168, 55.39106, 56.22905, 57.15084, ...
    58.32402, 59.32961, 60.41899, 61.34078, 62.34637, 63.43575, 64.44134, ...
    65.53073, 66.62011, 67.79330, 69.13408, 70.05587, 70.97765, 68.46369, ...
    71.98324, 73.24022, 74.41341, 75.50279, 76.84358, 77.68156, 78.60335, ...
    79.52514, 80.69832, 81.62011, 82.54190, 83.54749, 84.38547, 85.22346, ...
    86.06145, 87.06704, 88.07263, 88.82682, 76.17318, 72.65363, ...
];



y0 = 14.62396;             % Condición inicial

% Definición de la ecuación diferencial
f = @(t, y) 22.9811 + -0.186*y + -0.115*t + 0.173*sin(y) + -0.031*cos(y) + 0.578*sin(t) + -1.083*cos(t);

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
    14.62396, 17.13092, 19.84680, 22.35376, 25.27855, 27.99443, 30.91922, ...
    34.67967, 38.44011, 41.78273, 44.91643, 48.05014, 50.76602, 53.89972, ...
    57.03343, 59.74930, 63.09192, 65.59889, 69.15042, 71.86630, 74.37326, ...
    76.88022, 79.17827, 82.10306, 84.61003, 86.69916, 88.99721, 90.87744, ...
    92.75766, 94.01114, 95.89136, 97.56267, 98.60724, 99.44290, 100.27855, ...
    100.48747, 100.69638, 100.90529, 100.90529, 101.11421, 101.11421, ...
    100.90529, 100.69638, 100.69638, 100.69638, 100.48747, 100.48747, ...
    100.48747, 100.48747, 100.27855, 100.06964, 100.06964, 99.65181, ...
    99.44290, 99.44290, 99.02507, 98.81616, 98.60724, 98.39833, 98.18942, ...
    97.56267, 97.35376, 96.93593, 96.51811, 96.10028, 95.89136, 95.26462, ...
    94.84680, 94.42897, 94.01114, 93.38440, 92.75766, 92.13092, 91.50418, ...
    91.08635, 90.25070, 89.62396, 88.99721, 88.16156, 87.53482, 86.49025, ...
    85.86351, 85.02786, 86.90808, 84.40111, 83.56546, 82.72981, 81.68524, ...
    80.43175, 80.01393, 79.38719, 78.76045, 77.71588, 76.67131, 75.83565, ...
    75.20891, 74.37326, 73.53760, 72.91086, 71.86630, 70.82173, 69.98607, ...
    81.26741, 83.98329
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
nombreArchivo = 'Polinomios + sinusoides Sol. Númerica Number Facility (N).xlsx';

writetable(tabla_euler_explicito, nombreArchivo, 'Sheet', 'Euler_Explicito');
writetable(tabla_euler_implicito, nombreArchivo, 'Sheet', 'Euler_Implicito');
writetable(tabla_euler_mejorado, nombreArchivo, 'Sheet', 'Euler_Mejorado');
writetable(tabla_runge_kutta, nombreArchivo, 'Sheet', 'Runge_Kutta');
writetable(tabla_taylor, nombreArchivo, 'Sheet', 'Taylor_2do_Orden');