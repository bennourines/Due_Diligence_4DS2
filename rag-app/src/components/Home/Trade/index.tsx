import Image from "next/image";


const Trade = () => {
    return (
        <section>
            <div className="container mx-auto lg:max-w-screen-xl md:max-w-screen-md px-4 relative">
                <div className="bg-gradient-to-r from-primary to-secondary hidden lg:block absolute w-full h-full top-1/2 -right-1/4 blur-390"></div>
                <div className="grid lg:grid-cols-2 gap-x-5 relative z-10">
                    <div>
                        <Image src={'/images/trade/macbook.png'} alt="macBook-image" width={787} height={512} />
                    </div>
                    <div>
                        <h3 className="text-3xl lg:text-5xl font-semibold text-white mb-6 text-center sm:text-start">Trade Anywhere <br /> Any time</h3>
                        <p className="lg:text-lg font-normal text-white/40 mb-16 text-center sm:text-start">Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book.</p>
                        <div className="flex justify-between">
                            <Image src={'/images/trade/mac.svg'} alt="macOS-image" width={61} height={105} />
                            <div className="verticalLine"></div>
                            <Image src={'/images/trade/appstore.svg'} alt="appstore-image" width={80} height={105} />
                            <div className="verticalLine"></div>
                            <Image src={'/images/trade/windows.svg'} alt="windows-image" width={80} height={105} />
                            <div className="verticalLine"></div>
                            <Image src={'/images/trade/android.svg'} alt="android-image" width={71} height={105} />
                        </div>
                    </div>
                </div>
            </div>
        </section>
    )
}

export default Trade;
