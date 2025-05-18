"use client"
import React, { Component } from "react";
import Slider from "react-slick";
import "slick-carousel/slick/slick.css";
import "slick-carousel/slick/slick-theme.css";
import { Companiesdata } from "@/app/api/data";

// CAROUSEL SETTINGS
const Companies = () => {
    const settings = {
        dots: false,
        infinite: true,
        slidesToShow: 5,
        slidesToScroll: 1,
        arrows: false,
        autoplay: true,
        cssEase: "linear",
        responsive: [
            {
                breakpoint: 1024,
                settings: {
                    slidesToShow: 4,
                }
            },
            {
                breakpoint: 700,
                settings: {
                    slidesToShow: 2,
                }
            },
            {
                breakpoint: 500,
                settings: {
                    slidesToShow: 1,
                }
            }
        ]
    };

    return (
        <section className='text-center' >
            <div className="container mx-auto lg:max-w-screen-xl md:max-w-screen-md px-4">
                <div>
                    <Slider {...settings}>
                        {Companiesdata.map((item, i) =>
                            <div key={i}>
                                <img src={item.imgSrc} alt={item.imgSrc} />
                            </div>
                        )}
                    </Slider>
                </div>
            </div>
        </section>
    )
}

export default Companies;