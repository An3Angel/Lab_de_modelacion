%Perceptual speed
% Solución de una EDO usando métodos numéricos: Euler explícito, implícito,
% mejorado, Runge-Kutta de 4º orden y Taylor de 2º Orden

% Parámetros iniciales
t_inicial = 6.03352;       % Tiempo inicial en años
t_final = 89.58101;        % Tiempo final en años
n = 121;                   % Número de pasos
t_values = [6.03352, 6.20112, 6.28492, 6.53631, 6.62011, 6.78771, 6.95531, 7.12291, 7.29050, ...
            7.37430, 7.45810, 7.70950, 7.87709, 8.04469, 8.21229, 8.37989, 8.54749, 8.71508, 8.96648, ...
            9.05028, 9.21788, 9.46927, 9.72067, 9.88827, 9.97207, 10.30726, 10.39106, 10.55866, 10.89385, ...
            11.14525, 11.39665, 11.64804, 11.98324, 12.23464, 12.65363, 13.07263, 13.40782, 13.82682, ...
            14.41341, 15.00000, 15.50279, 16.08939, 16.75978, 17.51397, 18.18436, 19.10615, 20.02793, ...
            20.94972, 21.78771, 22.70950, 23.63128, 24.46927, 25.39106, 26.39665, 27.40223, 28.32402, ...
            29.41341, 30.50279, 31.67598, 32.76536, 33.77095, 34.86034, 35.94972, 37.03911, 38.29609, ...
            39.21788, 40.22346, 41.14525, 42.06704, 43.15642, 44.24581, 45.33520, 46.50838, 47.51397, ...
            48.68715, 49.69274, 50.69832, 51.62011, 52.54190, 53.63128, 54.63687, 55.72626, 56.64804, ...
            57.56983, 58.49162, 59.41341, 60.41899, 61.42458, 62.51397, 63.35196, 64.18994, 65.11173, ...
            65.94972, 66.95531, 67.87709, 68.71508, 69.63687, 70.47486, 71.39665, 72.23464, 73.07263, ...
            74.07821, 74.91620, 75.75419, 76.67598, 77.59777, 78.35196, 79.02235, 79.77654, 80.61453, ...
            81.36872, 82.20670, 83.04469, 83.79888, 84.72067, 85.55866, 86.31285, 87.15084, 87.90503, ...
            88.74302, 89.58101];
y0 = 17.33983;             % Condición inicial

% Definición de la ecuación diferencial
f = @(t, y) 30.029 - 0.193*y - 0.168*t + 0.496*sin(y) - 0.056*cos(y) - 0.103*sin(t) - 0.727*cos(t);

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
    17.33983, 20.05571, 22.56267, 25.27855, 28.41226, 31.12813, 34.26184, 36.7688, ...
    39.48468, 42.61838, 45.12535, 47.4234, 50.5571, 52.85515, 54.94429, 57.66017, 60.37604, ...
    62.88301, 65.59889, 68.31476, 70.4039, 72.70195, 75.62674, 77.92479, 80.01393, 82.10306, ...
    84.1922, 86.28134, 88.7883, 91.08635, 93.3844, 95.89136, 98.60724, 101.32312, 103.62117, ...
    105.91922, 108.42618, 110.72423, 113.02228, 115.32033, 117.40947, 119.08078, 120.54318, ...
    122.00557, 123.05014, 123.88579, 124.51253, 124.72145, 124.72145, 124.72145, 124.93036, ...
    124.93036, 124.72145, 124.72145, 124.72145, 124.72145, 124.51253, 124.30362, 124.30362, ...
    124.09471, 123.88579, 123.67688, 123.46797, 123.25905, 123.05014, 122.84123, 122.4234, ...
    122.4234, 122.21448, 121.79666, 121.37883, 120.961, 120.33426, 119.91643, 119.49861, ...
    119.08078, 118.66295, 118.24513, 117.8273, 116.99164, 116.57382, 115.73816, 114.90251, ...
    114.06685, 113.44011, 112.60446, 111.97772, 111.14206, 110.09749, 109.26184, 108.42618, ...
    107.59053, 106.75487, 105.91922, 104.87465, 103.83008, 102.99443, 101.94986, 100.90529, ...
    99.86072, 99.02507, 97.9805, 96.93593, 95.68245, 94.42897, 93.3844, 92.75766, 91.92201, ...
    90.87744, 89.41504, 88.57939, 87.53482, 86.49025, 85.23677, 83.98329, 83.14763, 81.89415, ...
    80.84958, 79.5961, 78.34262, 77.29805
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
nombreArchivo = 'Polinomios + sinusoides Sol. Númerica Perceptual speed.xlsx';

writetable(tabla_euler_explicito, nombreArchivo, 'Sheet', 'Euler_Explicito');
writetable(tabla_euler_implicito, nombreArchivo, 'Sheet', 'Euler_Implicito');
writetable(tabla_euler_mejorado, nombreArchivo, 'Sheet', 'Euler_Mejorado');
writetable(tabla_runge_kutta, nombreArchivo, 'Sheet', 'Runge_Kutta');
writetable(tabla_taylor, nombreArchivo, 'Sheet', 'Taylor_2do_Orden');

