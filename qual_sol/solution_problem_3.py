from typing import Tuple
import torch
from torch import Tensor

def t_matrix(
    dim: int,
    target: Tuple[int, int],
    theta: Tensor,
    phi: Tensor,
    dtype=torch.complex128,
    device='cpu'
) -> Tensor:
    """
    Матричная форма 2x2 произвольной унитарной матрицы, которая действует на выделенные 2 элемента вектора амплитуд (m, n).

    Блок параметризуется следующим образом:
        [[cos(theta),        -exp(-i*phi) * sin(theta)],
         [exp(i*phi) * sin(theta),  cos(theta)]]

    Параметры:
        dim: размер полной матрицы
        target: кортеж (m, n) — индексы амплитуд, на которые действует унитарное преобразование
        theta: угол вращения
        phi: фазовый угол
        dtype: тип данных
        device: устройство

   Вывод:
        Полная dim x dim унитарная матрица с заполненным блоком. 
    """
    m, n = target

    T = torch.eye(dim, dtype=dtype, device=device)

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    exp_iphi = torch.exp(1j * phi)

    # Заполняем 2x2 подматрицу
    T[m, m] = cos_t
    T[m, n] = -torch.conj(exp_iphi) * sin_t  # = -exp(-i*phi) * sin(theta)
    T[n, m] = exp_iphi * sin_t               # =  exp(i*phi) * sin(theta)
    T[n, n] = cos_t

    return T

def reck_decomposition(U: Tensor):
    """
    Декомпозиция Река на 2x2 блочные матрицы и диагональную матрицу фаз [1].
    
    Ввод:
        U: Матрица, которая будем расслад

    Вывод:
        thetas: Значения углов theta в каждом блоке
        phis:   Значения фаз phi в каждом блоке
        targets: Пары мод, между которыми ставиться блок
        phases: Диагональные фазосменщики
    
    [1] Experimental realization of any discrete unitary operator (1994), Reck et al.
    """
    N = U.shape[0]
    device = U.device
    dtype_real = torch.float64
    dtype_int = torch.int64

    thetas_list = []
    phis_list = []
    targets_list = []

    U_work = U.clone()

    for n in range(N - 1, 0, -1):          
        for m in range(n - 1, -1, -1):     
            a = U_work[n, n]
            b = U_work[m, n]

            r = torch.sqrt(torch.abs(a)**2 + torch.abs(b)**2 + 1e-15)

            if r < 1e-12:
                theta = torch.tensor(0.0, dtype=dtype_real, device=device)
                phi = torch.tensor(0.0, dtype=dtype_real, device=device)
            else:
                cos_theta = torch.abs(a) / r
                cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
                theta = torch.acos(cos_theta)
                phi = torch.angle(a) - torch.angle(b)

            thetas_list.append(theta)
            phis_list.append(phi)
            targets_list.append(torch.tensor([m, n], dtype=dtype_int, device=device))
            
            T_mn = t_matrix(N, target=(m,n), theta=theta, phi=phi)

            U_work = T_mn @ U_work

    thetas = torch.stack(thetas_list) if thetas_list else torch.empty(0, dtype=dtype_real, device=device)
    phis = torch.stack(phis_list) if phis_list else torch.empty(0, dtype=dtype_real, device=device)
    targets = torch.stack(targets_list) if targets_list else torch.empty((0, 2), dtype=dtype_int, device=device)

    phases = torch.angle(torch.diag(U_work)) 

    return thetas, phis, targets, phases

def get_matrix(
    dim: int, 
    phases: Tensor,
    phis: Tensor,
    thetas: Tensor,
    targets: Tensor,
    dtype=torch.complex128,
    device='cpu'
) -> Tensor:
    """
    Получение унитарной матрицы из известных параметров после декомпозиции Река. 

    Ввод:
        dim: размер матрицы
        phases: Диагональные фазосменщики
        phis:   Значения фаз phi в каждом блоке
        thetas: Значения углов theta в каждом блоке
        targets: Пары мод, между которыми ставиться блок
        dtype: тип данных
        device: устройство
    
    Вывод:
        U: Унитарная матрица dim x dim, полученная в результате декомпозиции Река
        layers: Словарь номер_слоя -> тензор блочной матрицы
    """
    U = torch.diag(torch.exp(1j * phases.to(dtype))).to(device)
    layers = dict.fromkeys([i for i in range(dim*(dim - 1)//2 + 1)])
    layers[0] = U.clone()
    for i in range(len(thetas) - 1, -1, -1):
        theta = thetas[i]
        phi = phis[i]
        m, n = targets[i].tolist()
        T_mn = t_matrix(dim, target=(m,n), theta=theta, phi=phi)
        U = T_mn.adjoint() @ U
        layers[i+1] = T_mn.adjoint()
    return U, layers

def fidellity(U_exp: Tensor, U: Tensor):
    """
    Мера близости между двумя унитарными матрицами

    Fid = |Tr(U_{exp}^{dagger} U)|/N , где N - размерность матриц
    """
    return torch.abs(torch.einsum('ii', U_exp.adjoint() @ U)).item()/U.shape[0]

def haar_measure(N):
    """
    Генератор Хааровских матриц с помощью QR декомпозициия
    """
    A, B = torch.randn(size=(N, N), dtype=torch.float64), torch.randn(size=(N, N), dtype=torch.float64)
    Z = A + 1j * B
    Q, R = torch.linalg.qr(Z)
    Lambda = torch.diag(torch.tensor([R[i, i] / torch.abs(R[i, i]) for i in range(N)]))

    return (Q @ Lambda).numpy()

if __name__ == '__main__':
    torch.manual_seed(42)
    N = torch.randint(low=2, high=10, size = ())

    #принимаем на вход случайную унитарную матрицу в формате numpy
    U = haar_measure(N)

    #Переводим в torch,tensor()
    U = torch.tensor(U)
    print(f"Случайная унитарная матрица размера {N} из распределения Хаара :\n {U.numpy().round(4)}")

    #Проводим разложение Река -> получаем параметры
    thetas, phis, targets, phases = reck_decomposition(U)

    #Восстанавливаем матрицу -> получаем матрицу + словарь с номером слоя и его матрицей
    reconstructed_unitary, layers = get_matrix(N, phases, phis, thetas, targets)

    print("\nВосстановленая с помощью декомпозиции Река унитарная матрица:\n", reconstructed_unitary.detach().numpy().round(4))
    print("\nФиделити:\n", fidellity(reconstructed_unitary, U))
    assert torch.allclose(U, reconstructed_unitary, atol=1e-06), "Восстановленая матрица не совпадает с входной"

    print("\nМатрицы для каждого слоя\n")
    for i in layers.keys():
        theta = thetas[i-1]
        phi = phis[i-1]
        m, n = targets[i-1].tolist()
        print(f'Номер слоя {i}: \t\t Параметры: theta: {theta:.3f}, phi: {phi:.2f}, target: {(m, n)} \n {layers[i]}')