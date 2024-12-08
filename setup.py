from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='drug_molecule_generation_with_vae',
    version='0.1.0',
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'drug_molecule_generation_with_vae=drug_molecule_generation_with_vae.cli:cli',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for generating drug molecules using a Variational Autoencoder (VAE)',
    url='https://github.com/yourusername/drug_molecule_generation_with_vae',  # Update with your repository URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.12',
)
