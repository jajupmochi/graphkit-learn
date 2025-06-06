"""
generate_reference_md

@Author: jajupmochi
@Date: Jun 06 2025
"""
import os
import shutil
from datetime import datetime
import yaml

PACKAGE_NAME = "gklearn"
SOURCE_DIR = os.path.join(os.path.dirname(__file__), PACKAGE_NAME)
DOCS_DIR = "docs"
REFERENCE_DIR = os.path.join(DOCS_DIR, "reference")

# Directories to exclude, relative to PACKAGE_NAME
EXCLUDE_DIRS = {
	"examples",
	"tests",
	os.path.join("gedlib", "include"),
}

# Module names to exclude explicitly
EXCLUDE_MODULES = {
	f"{PACKAGE_NAME}.__init__",  # 明确排除 __init__ 模块
}


def should_exclude_dir(root):
	"""
	Decide if a directory should be excluded based on its relative path.
	Also exclude any directory that contains 'experiments' in its path parts.
	"""
	rel = os.path.relpath(root, SOURCE_DIR)
	parts = rel.split(os.sep)
	if rel == ".":
		return False

	# Exclude any folder with 'experiments' in the path parts (recursive)
	if "experiments" in parts:
		return True

	# Exclude directories in EXCLUDE_DIRS (only top-level relative to PACKAGE_NAME)
	for ex in EXCLUDE_DIRS:
		ex_parts = ex.split(os.sep)
		if parts[:len(ex_parts)] == ex_parts:
			return True

	# Special handling for gklearn/gedlib:
	# Only include gedlib root and gedlib/src folder, exclude others
	if parts[0] == "gedlib":
		if len(parts) == 1:
			return False  # Include gedlib root
		if len(parts) == 2 and parts[1] == "src":
			return False  # Include gedlib/src
		return True  # Exclude other subdirs in gedlib

	return False


def is_valid_python_module(filepath):
	"""
	Check if a file is a valid Python module that should be documented.
	Exclude __init__.py and non-standard Python files.
	"""
	filename = os.path.basename(filepath)

	# 排除 __init__.py 文件
	if filename == "__init__.py":
		return False

	# 只包含 .py 文件，排除 .pyx 和 .pxd
	if not filename.endswith(".py"):
		return False

	# 排除以下划线开头的私有模块（除了特殊模块）
	if filename.startswith("_") and not filename.startswith("__"):
		return False

	return True


def find_modules(root):
	"""
	Recursively find all Python modules (.py) under root, excluding
	specified directories and modules.
	Returns a dictionary with module info for better organization.
	"""
	modules = []
	for dirpath, dirnames, filenames in os.walk(root):
		if should_exclude_dir(dirpath):
			# Skip excluded directories entirely
			dirnames.clear()
			continue

		for f in filenames:
			filepath = os.path.join(dirpath, f)

			# 使用改进的模块验证函数
			if is_valid_python_module(filepath):
				rel_path = os.path.relpath(filepath, SOURCE_DIR)
				mod_path = rel_path.replace(os.sep, ".")
				mod_name = mod_path.rsplit(".", 1)[0]  # Remove file extension
				full_mod = f"{PACKAGE_NAME}.{mod_name}"

				if full_mod in EXCLUDE_MODULES:
					continue

				# 计算相对于 gklearn 的路径层级
				path_parts = rel_path.split(os.sep)[:-1]  # 去掉文件名
				category = path_parts[0] if path_parts else "core"

				modules.append(
					{
						'full_name': full_mod,
						'short_name': mod_name.split('.')[-1],
						'category': category,
						'path_parts': path_parts,
						'file_path': rel_path
					}
				)

	return sorted(modules, key=lambda x: x['full_name'])


def backup_existing_docs(reference_dir):
	"""
	Backup existing reference documentation directory.
	"""
	if os.path.exists(reference_dir):
		timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
		backup_path = f"{reference_dir}.{timestamp}.bak"
		shutil.copytree(reference_dir, backup_path)
		print(f"Backed up existing '{reference_dir}' to '{backup_path}'")
		shutil.rmtree(reference_dir)


def create_module_page(module_info, output_dir):
	"""
	Create individual markdown page for a module.
	"""
	full_name = module_info['full_name']
	short_name = module_info['short_name']
	category = module_info['category']

	# 创建文件路径
	category_dir = os.path.join(output_dir, category)
	os.makedirs(category_dir, exist_ok=True)

	filename = f"{short_name}.md"
	filepath = os.path.join(category_dir, filename)

	with open(filepath, "w", encoding="utf-8") as f:
		f.write(f"# {short_name}\n\n")
		f.write(f"Module: `{full_name}`\n\n")
		f.write(f"::: {full_name}\n")

	return {
		'title': short_name,
		'path': os.path.relpath(filepath, output_dir).replace(os.sep, '/'),
		'category': category
	}


def generate_nav_structure(page_info_list):
	"""
	Generate navigation structure for mkdocs.yml
	"""
	nav_structure = {}

	for page_info in page_info_list:
		category = page_info['category']
		if category not in nav_structure:
			nav_structure[category] = []

		nav_structure[category].append(
			{
				'title': page_info['title'],
				'path': f"reference/{page_info['path']}"
			}
		)

	return nav_structure


def create_index_page(nav_structure, output_dir):
	"""
	Create an index page for the reference documentation.
	"""
	index_path = os.path.join(output_dir, "index.md")

	with open(index_path, "w", encoding="utf-8") as f:
		f.write("# API Reference\n\n")
		f.write("Welcome to the gklearn API reference documentation.\n\n")

		for category, pages in nav_structure.items():
			f.write(f"## {category.title()}\n\n")
			for page in pages:
				f.write(f"- [{page['title']}]({page['path']})\n")
			f.write("\n")


def update_mkdocs_nav(nav_structure, mkdocs_path="mkdocs.yml"):
	"""
	Update or create mkdocs.yml with the new navigation structure.
	"""
	# 构建导航结构
	reference_nav = []
	reference_nav.append("reference/index.md")

	for category, pages in sorted(nav_structure.items()):
		category_section = {category.title(): []}
		for page in sorted(pages, key=lambda x: x['title']):
			category_section[category.title()].append(
				{
					page['title']: page['path']
				}
			)
		reference_nav.append(category_section)

	# 读取现有的 mkdocs.yml 或创建新的
	config = {}
	if os.path.exists(mkdocs_path):
		try:
			with open(mkdocs_path, 'r', encoding='utf-8') as f:
				config = yaml.safe_load(f) or {}
		except Exception as e:
			print(f"Warning: Could not read existing mkdocs.yml: {e}")

	# 更新导航
	if 'nav' not in config:
		config['nav'] = []

	# 移除旧的 API Reference 部分
	config['nav'] = [item for item in config['nav']
	                 if not (isinstance(item, dict) and 'API Reference' in item)]

	# 添加新的 API Reference 结构
	config['nav'].append({'API Reference': reference_nav})

	# 确保基本配置存在
	if 'site_name' not in config:
		config['site_name'] = 'gklearn Documentation'

	if 'plugins' not in config:
		config['plugins'] = ['mkdocstrings']
	elif 'mkdocstrings' not in config['plugins']:
		config['plugins'].append('mkdocstrings')

	# 写回 mkdocs.yml
	with open(mkdocs_path, 'w', encoding='utf-8') as f:
		yaml.dump(config, f, default_flow_style=False, sort_keys=False)

	print(f"Updated navigation structure in '{mkdocs_path}'")


def main():
	"""
	Main function to generate reference documentation.
	"""
	# 确保输出目录存在
	os.makedirs(DOCS_DIR, exist_ok=True)

	# 备份现有文档
	backup_existing_docs(REFERENCE_DIR)

	# 创建新的输出目录
	os.makedirs(REFERENCE_DIR, exist_ok=True)

	# 查找所有模块
	modules = find_modules(SOURCE_DIR)
	print(f"Found {len(modules)} modules to document")

	# 为每个模块创建页面
	page_info_list = []
	for module_info in modules:
		page_info = create_module_page(module_info, REFERENCE_DIR)
		page_info_list.append(page_info)
		print(f"Created documentation for: {module_info['full_name']}")

	# 生成导航结构
	nav_structure = generate_nav_structure(page_info_list)

	# 创建索引页面
	create_index_page(nav_structure, REFERENCE_DIR)

	# 更新 mkdocs.yml
	update_mkdocs_nav(nav_structure)

	print(f"\nDocumentation generation completed!")
	print(f"- Generated {len(page_info_list)} module pages")
	print(f"- Organized into {len(nav_structure)} categories")
	print(f"- Excluded modules: {EXCLUDE_MODULES}")
	print(f"- Documentation saved to: {REFERENCE_DIR}")


if __name__ == "__main__":
	main()