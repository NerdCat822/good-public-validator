import Image from "next/image";
import React from "react";

const Loading = () => {
  return (
    <div className="fixed left-0 top-0 flex size-full items-center justify-center bg-black/50">
      <Image src="/images/loading.gif" alt="Loading" width={50} height={50} />
    </div>
  );
};

export default Loading;
