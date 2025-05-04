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


def jacobi_svd(matrix, tol=1e-6):
    """Метод вращений Якоби для SVD"""
    n, m = matrix.shape
    U = np.eye(n)
    V = np.eye(m)
    S = matrix.copy()
    
    for _ in range(100):  # Максимальное число итераций
        # Находим максимальный внедиагональный элемент
        max_val = 0
        p, q = 0, 0
        for i in range(min(n, m)):
            for j in range(i+1, min(n, m)):
                if abs(S[i,j]) > max_val:
                    max_val = abs(S[i,j])
                    p, q = i, j
        
        if max_val < tol:
            break
            
        # Вычисляем угол вращения
        if S[p,p] == S[q,q]:
            theta = np.pi/4
        else:
            theta = 0.5 * np.arctan(2*S[p,q] / (S[p,p] - S[q,q]))
        
        # Матрица вращения
        J = np.eye(min(n, m))
        J[p,p] = np.cos(theta)
        J[q,q] = np.cos(theta)
        J[p,q]

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
