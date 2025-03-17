<!-- Improved compatibility of back to top link: See: https://github.com/sirmelonchen/HackITM2025/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Issues][issues-shield]][issues-url]
[![Unlicense License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/sirmelonchen/HackITM2025">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">HackITM 2025</h3>

  <p align="center">
    Can an AI upscale old videomaterials?
    <br />
    <a href="https://github.com/sirmelonchen/HackITM2025/"><strong>Explore the docs »</strong></a>
    <br />
    &middot;
    <a href="https://github.com/sirmelonchen/HackITM2025/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    &middot;
    <a href="https://github.com/sirmelonchen/HackITM2025/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

With the help of AI models such as Real-ESRGAN, old TV recordings are automatically enhanced and scaled to higher resolutions, e.g. from SD to HD or 4K. The process uses a powerful GPU to process images and videos in real time or near real time. This makes it possible, for example, to reuse historical material for modern broadcasts or archive projects in impressive quality.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

_For this script to run you need to install python3.

* Python:
    Version: [3.11.0](https://www.python.org/downloads/release/python-3110/)

### Installation

1. Clone the repo

   ```sh
   git clone https://github.com/sirmelonchen/HackITM2025
   ```

2. Install Python packages

   ```sh
   pip install -r requirements.txt
   ```

3. Edit the video path in `script.py`

   ```py
   video_path = "short.mp4"
   ```

4. Execute the Script in the same folder as the model and video

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

You could use the program to upscale old image material (still without sound) without loss. This could be useful for TV studios or influencers.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ROADMAP -->
## Roadmap

* [ ] Add Changelog
* [x] Add back to top links
* [ ] Add Additional Templates w/ Examples
* [ ] Add "components" document to easily copy & paste sections of the readme

See the [open issues](https://github.com/sirmelonchen/HackITM2025/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


<!-- LICENSE -->
## License

Distributed under the Unlicense License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
## Contact

Your Name - [@sirmelonchen](https://github.com/sirmelonchen)

Project Link: [https://github.com/sirmelonchen/HackITM2025](https://github.com/sirmelonchen/HackITM2025)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/sirmelonchen/HackITM2025.svg?style=for-the-badge
[contributors-url]: https://github.com/sirmelonchen/HackITM2025/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/sirmelonchen/HackITM2025.svg?style=for-the-badge
[forks-url]: https://github.com/sirmelonchen/HackITM2025/network/members
[stars-shield]: https://img.shields.io/github/stars/sirmelonchen/HackITM2025.svg?style=for-the-badge
[stars-url]: https://github.com/sirmelonchen/HackITM2025/stargazers
[issues-shield]: https://img.shields.io/github/issues/sirmelonchen/HackITM2025.svg?style=for-the-badge
[issues-url]: https://github.com/sirmelonchen/HackITM2025/issues
[license-shield]: https://img.shields.io/github/license/sirmelonchen/HackITM2025.svg?style=for-the-badge
[license-url]: https://github.com/sirmelonchen/HackITM2025/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com
