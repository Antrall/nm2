import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image

# Загрузка изображения
img = image.imread('Ориентальные-котята-5.jpg')

# Отображение исходного изображения
plt.imshow(img)
plt.title('Исходное изображение')
plt.show()

# Разделение на цветовые каналы
R, G, B = img[:,:,0]/255.0, img[:,:,1]/255.0, img[:,:,2]/255.0


def power_method_svd(matrix, max_iter=100, tol=1e-6):
    """Степенной метод для нахождения сингулярных значений и векторов"""
    n, m = matrix.shape
    svd_results = []
    
    # Копируем матрицу для работы
    M = matrix.copy()
    
    for _ in range(min(n, m)):
        # Начальное приближение - случайный вектор
        x = np.random.randn(min(n, m))
        x = x / np.linalg.norm(x)
        
        for _ in range(max_iter):
            # Итерация степенного метода
            x_new = M.T @ (M @ x)
            x_new = x_new / np.linalg.norm(x_new)
            
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        
        # Вычисляем сингулярное значение
        sigma = np.linalg.norm(M @ x)
        
        # Вычисляем левый сингулярный вектор
        u = (M @ x) / sigma
        
        # Сохраняем результаты
        svd_results.append((sigma, u, x))
        
        # Дефляция матрицы
        M = M - sigma * np.outer(u, x)
    
    return svd_results

# Применяем степенной метод для каждого канала с разным числом итераций
iterations = [10, 50, 100]
results = {}

for channel, name in zip([R, G, B], ['R', 'G', 'B']):
    channel_results = {}
    for n_iter in iterations:
        svd_result = power_method_svd(channel, max_iter=n_iter)
        # Восстанавливаем матрицу из SVD
        reconstructed = np.zeros_like(channel)
        for sigma, u, v in svd_result:
            reconstructed += sigma * np.outer(u, v)
        # Вычисляем норму разности
        error = np.linalg.norm(channel - reconstructed)
        channel_results[n_iter] = (svd_result, error)
    results[name] = channel_results

# Выводим результаты ошибок
for channel_name in results:
    print(f"\nКанал {channel_name}:")
    for n_iter in iterations:
        error = results[channel_name][n_iter][1]
        print(f"Итераций: {n_iter}, Ошибка: {error:.6f}")


# Функция для усеченного восстановления
def truncated_reconstruction(svd_result, k):
    """Восстановление изображения с использованием k сингулярных значений"""
    reconstructed = np.zeros_like(R)
    for i in range(k):
        sigma, u, v = svd_result[i]
        reconstructed += sigma * np.outer(u, v)
    return np.clip(reconstructed, 0, 1)

# Анализ для канала R (аналогично можно для G и B)
svd_r = power_method_svd(R, max_iter=100)
singular_values_r = [sigma for sigma, _, _ in svd_r]

# График сингулярных значений
plt.figure(figsize=(10, 5))
plt.plot(singular_values_r, 'o-')
plt.title('Сингулярные значения канала R')
plt.xlabel('Номер сингулярного значения')
plt.ylabel('Величина')
plt.grid()
plt.show()

# Визуализация восстановления с разным числом сингулярных значений
k_values = [1, 5, 10, 20, 50, 100]
plt.figure(figsize=(15, 10))
for i, k in enumerate(k_values):
    reconstructed = truncated_reconstruction(svd_r, k)
    plt.subplot(2, 3, i+1)
    plt.imshow(reconstructed, cmap='Reds')
    plt.title(f'k = {k}')
    plt.axis('off')
plt.suptitle('Восстановление канала R с разным числом сингулярных значений')
plt.show()



def jacobi_method(A, tolerance=1e-4, max_iterations=1000):
    """ Метод Якоби """
    # Проверка симметричности матрицы
    if not np.allclose(A, A.T):
        raise ValueError("Input matrix must be symmetric")
        
    n = A.shape[0]
    A = A.copy().astype(float)
    error_history = []
    
    # Начальная ошибка
    current_error = np.linalg.norm(np.triu(A, k=1))
    error_history.append(current_error)
    
    iteration = 0
    while current_error > tolerance and iteration < max_iterations:
        # Поиск максимального внедиагонального элемента
        upper_tri = np.triu(np.abs(A), k=1)
        p, q = np.unravel_index(np.argmax(upper_tri), A.shape)
        max_val = upper_tri[p, q]
        
        if max_val < 1e-12:  # Прерывание если элементы слишком малы
            break
            
        # Вычисление параметров вращения
        if A[p, p] == A[q, q]:
            # Обработка равных диагональных элементов
            cos_phi = np.sqrt(0.5)
            sin_phi = np.sqrt(0.5) * np.sign(A[p, q])
        else:
            t = (2 * A[p, q]) / (A[p, p] - A[q, q])
            cos_phi = np.sqrt(0.5 * (1 + 1 / np.sqrt(1 + t**2)))
            sin_phi = np.sign(t) * np.sqrt(0.5 * (1 - 1 / np.sqrt(1 + t**2)))
        
        # Создание матрицы вращения
        S = np.eye(n)
        S[p, p] = S[q, q] = cos_phi
        S[p, q] = -sin_phi
        S[q, p] = sin_phi
        
        # Применение вращения
        A = S.T @ A @ S
        
        # Обновление ошибки
        current_error = np.linalg.norm(np.triu(A, k=1))
        error_history.append(current_error)
        iteration += 1
    
    eigenvalues = np.diag(A)
    return eigenvalues, error_history

# Гистограммы для каждого канала
plt.figure(figsize=(15, 5))
for i, (channel, name) in enumerate(zip([R, G, B], ['Red', 'Green', 'Blue'])):
    plt.subplot(1, 3, i+1)
    plt.hist(channel.flatten(), bins=50, color=name.lower())
    plt.title(f'Гистограмма {name} канала')
    plt.xlabel('Значение пикселя')
    plt.ylabel('Частота')
plt.show()

# Параметры распределений
for channel, name in zip([R, G, B], ['Red', 'Green', 'Blue']):
    flat = channel.flatten()
    print(f"\n{name} канал:")
    print(f"Среднее: {np.mean(flat):.4f}")
    print(f"Стандартное отклонение: {np.std(flat):.4f}")
    print(f"Минимальное значение: {np.min(flat):.4f}")
    print(f"Максимальное значение: {np.max(flat):.4f}")

# Сравнение с числом значимых сингулярных значений
significant_sv = sum(sv > 0.1 * max(singular_values_r) for sv in singular_values_r)
print(f"\nЧисло значимых сингулярных значений (R канал): {significant_sv}")
