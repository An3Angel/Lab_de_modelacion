%Vocabulary
% Solución de una EDO usando métodos numéricos: Euler explícito, implícito,
% mejorado, Runge-Kutta de 4º orden y Taylor de 2º Orden

% Parámetros iniciales
t_inicial =   6.03352;       % Tiempo inicial en años
t_final = 89.74860;        % Tiempo final en años
n = 86;                   % Número de pasos
t_values = [
    6.03352, 6.45251, 6.70391, 7.12291, 7.45810, 7.79330, 8.21229, ...
    8.63128, 8.88268, 9.30168, 9.72067, 10.22346, 10.81006, 11.39665, ...
    12.31844, 13.32402, 11.89944, 14.07821, 12.82123, 14.74860, 15.58659, ...
    16.42458, 17.26257, 18.18436, 19.18994, 20.27933, 21.36872, 22.62570, ...
    23.46369, 24.55307, 25.64246, 26.81564, 27.82123, 29.24581, 30.50279, ...
    31.59218, 32.76536, 33.93855, 35.19553, 36.36872, 37.87709, 39.21788, ...
    40.39106, 41.64804, 42.98883, 44.32961, 45.50279, 46.67598, 47.68156, ...
    48.85475, 50.02793, 51.45251, 52.70950, 53.79888, 55.05587, 56.48045, ...
    57.56983, 58.32402, 59.16201, 60.08380, 61.08939, 62.09497, 63.10056, ...
    64.10615, 65.53073, 66.87151, 68.12849, 69.38547, 70.55866, 71.98324, ...
    73.32402, 74.58101, 75.75419, 77.09497, 78.26816, 79.27374, 80.27933, ...
    81.28492, 82.37430, 83.37989, 84.38547, 85.64246, 86.64804, 87.73743, ...
    88.82682, 89.74860
];




y0 = 3.55153;             % Condición inicial

% Definición de la ecuación diferencial
f = @(t, y) 8.2601 + -0.124*y + -0.027*t + -0.043*sin(y0) + 0.248*cos(y0) + 0.086*sin(t) + -0.154*cos(t);

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
    3.55153, 6.05850, 8.35655, 11.07242, 13.57939, 15.66852, 17.75766, ...
    19.84680, 21.72702, 23.81616, 26.11421, 28.41226, 30.91922, 33.42618, ...
    36.55989, 39.48468, 34.88858, 41.57382, 38.02228, 43.24513, 44.91643, ...
    46.37883, 47.84123, 49.09471, 50.34819, 51.18384, 51.81058, 52.64624, ...
    53.06407, 53.69081, 54.10864, 54.52646, 54.94429, 55.15320, 55.36212, ...
    55.77994, 55.98886, 56.19777, 56.40669, 56.61560, 56.82451, 57.03343, ...
    57.24234, 57.45125, 57.45125, 57.45125, 57.45125, 57.66017, 57.66017, ...
    57.66017, 57.86908, 57.66017, 57.66017, 57.66017, 57.66017, 57.66017, ...
    57.24234, 57.66017, 57.66017, 57.45125, 57.45125, 57.24234, 57.24234, ...
    57.03343, 57.03343, 56.82451, 56.40669, 55.98886, 55.77994, 55.77994, ...
    55.36212, 54.94429, 54.52646, 53.89972, 53.69081, 53.06407, 52.64624, ...
    52.22841, 51.60167, 51.39276, 50.55710, 49.93036, 49.51253, 48.67688, ...
    48.05014, 47.42340
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
nombreArchivo = 'Polinomios + sinusoides Sol. Númerica Vocabulary.xlsx';

writetable(tabla_euler_explicito, nombreArchivo, 'Sheet', 'Euler_Explicito');
writetable(tabla_euler_implicito, nombreArchivo, 'Sheet', 'Euler_Implicito');
writetable(tabla_euler_mejorado, nombreArchivo, 'Sheet', 'Euler_Mejorado');
writetable(tabla_runge_kutta, nombreArchivo, 'Sheet', 'Runge_Kutta');
writetable(tabla_taylor, nombreArchivo, 'Sheet', 'Taylor_2do_Orden');