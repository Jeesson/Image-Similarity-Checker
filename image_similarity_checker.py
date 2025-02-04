import os
import sys
import time
import cv2
import numpy as np
import platform
from datetime import timedelta
from skimage.metrics import structural_similarity as ssim
from math import comb
import ctypes


try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def set_window_title(title):
    """Изменяет заголовок окна консоли"""
    if platform.system() == 'Windows':
        ctypes.windll.kernel32.SetConsoleTitleW(title)
    elif platform.system() == 'Linux':
        sys.stdout.write(f"\x1b]2;{title}\x07")
    else:
        print(f"\33]0;{title}\a", end='', flush=True)


def print_progress(current, total, prefix=""):
    """Текстовый прогресс-бар"""
    bar_length = 40
    progress = current / total
    filled = int(bar_length * progress)
    bar = '█' * filled + '-' * (bar_length - filled)
    percent = progress * 100
    sys.stdout.write(f"\r{prefix} |{bar}| {percent:.1f}% ({current}/{total})")
    sys.stdout.flush()


def load_image(image_path, verbose=True):
    """Загружает изображение с ограниченным выводом истории"""
    absolute_path = os.path.abspath(image_path)
    try:
        with open(absolute_path, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if verbose and not tqdm:
            if not hasattr(load_image, "history"):
                load_image.history = []
            load_image.history.append(os.path.basename(image_path)[:15])
            load_image.history = load_image.history[-10:]

            progress = f"Последние файлы: {', '.join(load_image.history)}"
            sys.stdout.write(f"\r{progress.ljust(120)}")
            sys.stdout.flush()

        return image
    except Exception as e:
        print(f"\nОшибка загрузки {os.path.basename(image_path)}: {e}")
        return None


def preprocess_image(image, size=(300, 300)):
    """Обработка изображения"""
    image = cv2.resize(image, size)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def compare_images(img1, img2):
    """Сравнение изображений"""
    score, _ = ssim(img1, img2, full=True)
    return score * 100


def compare_two_images():
    """Сравнение двух изображений"""
    image1_path = input("Введите путь к первому изображению: ").strip()
    image2_path = input("Введите путь ко второму изображению: ").strip()

    img1 = load_image(image1_path, verbose=False)
    img2 = load_image(image2_path, verbose=False)

    img1_proc = preprocess_image(img1)
    img2_proc = preprocess_image(img2)

    if img1_proc.shape != img2_proc.shape:
        img2_proc = cv2.resize(img2_proc, (img1_proc.shape[1], img1_proc.shape[0]))

    similarity = compare_images(img1_proc, img2_proc)
    print(f"\nСхожесть: {similarity:.2f}%")

def compare_image_with_folder(base_image_path, folder_path, min_similarity):
    """Сравнение с изображениями в папке"""
    base_image = load_image(base_image_path, verbose=False)
    base_image_proc = preprocess_image(base_image)

    results = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.abspath(file_path) == os.path.abspath(base_image_path):
            continue

        try:
            compare_img = load_image(file_path, verbose=False)
            compare_img_proc = preprocess_image(compare_img)

            if base_image_proc.shape != compare_img_proc.shape:
                compare_img_proc = cv2.resize(compare_img_proc, 
                    (base_image_proc.shape[1], base_image_proc.shape[0]))

            similarity = compare_images(base_image_proc, compare_img_proc)
            if similarity >= min_similarity:
                results.append((filename, similarity))
                print(f"{filename}: {similarity:.2f}%")
        except:
            continue

    if results:
        results.sort(key=lambda x: x[1], reverse=True)
        print("\nРезультаты:")
        for filename, similarity in results:
            print(f"{filename}: {similarity:.2f}%")
    else:
        print("Совпадений не найдено.")

def auto_scan_folder(folder_path, min_similarity):
    """Авто-скан всех изображений в папке"""
    try:
        if not os.path.isdir(folder_path):
            print(f"Ошибка: {folder_path} не существует")
            return []

        image_files = [
            os.path.join(folder_path, f) for f in os.listdir(folder_path) 
            if f.lower().endswith(('.png','.jpg','.jpeg','.bmp','.tiff'))
        ]

        if len(image_files) < 2:
            print("Нужно минимум 2 изображения")
            return []

        total_pairs = comb(len(image_files), 2)
        results = []
        start_time = time.time()

        # Прогресс-бар загрузки
        if tqdm:
            load_pbar = tqdm(image_files, desc="🔄 Загрузка", unit="file")
        else:
            print("Загрузка изображений...")

        # Предзагрузка изображений
        processed_images = {}
        for img_path in image_files:
            try:
                img = load_image(img_path, verbose=not tqdm)
                processed = preprocess_image(img)
                processed_images[img_path] = processed
                if tqdm:
                    load_pbar.update(1)
                    load_pbar.set_postfix(file=os.path.basename(img_path)[:20])
            except Exception as e:
                print(f"\nОшибка: {os.path.basename(img_path)} - {str(e)}")
                continue

        if tqdm:
            load_pbar.close()

        # Основное сравнение
        if tqdm:
            pbar = tqdm(total=total_pairs, desc="🔍 Сравнение", unit="pair", 
                        bar_format="{l_bar}{bar:40}{r_bar}{bar:-40b}")
        else:
            print("\nНачато сравнение...")
            set_window_title("Авто-скан 0.0%")

        current_pair = 0
        for i in range(len(image_files)):
            img1_path = image_files[i]
            if img1_path not in processed_images:
                continue

            img1 = processed_images[img1_path]

            for j in range(i+1, len(image_files)):
                img2_path = image_files[j]
                if img2_path not in processed_images:
                    continue

                img2 = processed_images[img2_path]

                try:
                    if img1.shape != img2.shape:
                        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

                    similarity = compare_images(img1, img2)
                    if similarity >= min_similarity:
                        results.append((
                            os.path.basename(img1_path),
                            os.path.basename(img2_path),
                            similarity
                        ))
                except Exception as e:
                    print(f"\nОшибка сравнения: {e}")
                    similarity = 0

                # Обновление прогресса
                current_pair += 1
                if tqdm:
                    pbar.update(1)
                    pbar.set_postfix_str(
                        f"{os.path.basename(img1_path)[:15]} ↔ "
                        f"{os.path.basename(img2_path)[:15]} "
                        f"({similarity:.1f}%)"
                    )
                    progress = current_pair / total_pairs * 100
                    set_window_title(f"Авто-скан {progress:.1f}%")
                else:
                    set_window_title(f"Авто-скан {progress:.1f}%")
                    if current_pair % 10 == 0:
                        print_progress(current_pair, total_pairs)

        # Завершение
        if tqdm:
            pbar.close()
        else:
            sys.stdout.write("\n")
            sys.stdout.flush()

        # Вывод результатов
        print(f"\n{'='*50}")
        print(f"Найдено совпадений: {len(results)}")
        print(f"Время выполнения: {timedelta(seconds=int(time.time()-start_time))}")
        print(f"{'='*50}")

        results.sort(key=lambda x: x[2], reverse=True)
        for idx, (img1, img2, sim) in enumerate(results[:20], 1):
            print(f"{idx:3d}. {img1:25} ↔ {img2:25}: {sim:.2f}%")

        return results

    except KeyboardInterrupt:
        print("\n\nСканирование прервано!")
        return []
    finally:
        set_window_title("Готово")
        if not tqdm:
            sys.stdout.write("\r" + " " * 120 + "\r")  # Очистка строки


def main():
    print("Image Similarity Checker")
    print("Выберите режим:")
    print("1 - Быстрый режим (test.png в текущей папке)")
    print("2 - Расширенный режим")

    mode = input("Ваш выбор (1/2): ").strip()

    if mode == '1':
        base_path = os.path.join(os.getcwd(), "test.png")
        folder = os.getcwd()
        try:
            min_sim = float(input("Минимальный процент схожести: "))
            compare_image_with_folder(base_path, folder, min_sim)
        except:
            print("Некорректный ввод!")

    elif mode == '2':
        print("\nРежимы работы:")
        print("1 - Сравнить два изображения")
        print("2 - Сравнить с изображениями в папке")
        print("3 - Авто-скан всей папки")

        choice = input("Выберите действие (1-3): ").strip()

        if choice == '1':
            compare_two_images()
        elif choice == '2':
            base = input("Путь к базовому изображению: ").strip()
            folder = input("Путь к папке: ").strip()
            try:
                min_sim = float(input("Минимальный процент схожести: "))
                compare_image_with_folder(base, folder, min_sim)
            except:
                print("Ошибка ввода!")
        elif choice == '3':
            folder = input("Путь к папке: ").strip()
            try:
                min_sim = float(input("Минимальный процент схожести: "))
                auto_scan_folder(folder, min_sim)
            except:
                print("Ошибка ввода!")
        else:
            print("Неверный выбор")
    else:
        print("Неверный режим")

    input("\nНажмите Enter для выхода...")


if __name__ == "__main__":
    main()